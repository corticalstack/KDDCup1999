"""
===========================================================================
Multi Layer Perceptron - Binary
===========================================================================
Multi Layer Perceptron - Binary
"""
import os
import sys
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from keras import models, layers
from keras import optimizers
import keras.backend as K
from filehandler import Filehandler
from dataset import KDDCup1999
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import talos as ta



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class AnnMLPBinary:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore low level instruction warnings
        tf.logging.set_verbosity(tf.logging.ERROR)  # Set tensorflow verbosity
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # self.logfile = None
        # self.gettrace = getattr(sys, 'gettrace', None)
        # self.original_stdout = sys.stdout
        # self.timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.log_file()

        print(__doc__)

        self.random_state = 20
        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.visualize = Visualize()
        self.folder = 'viz'

        # Datasets
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_features = None
        self.label_map_int_2_string = {0: 'good', 1: 'bad', '0': 'good', '1': 'bad'}
        self.label_map_string_2_int = {'normal': 0, 'dos': 1, 'u2r': 1, 'r2l': 1, 'probe': 1}

        # K-fold validation
        self.splits = 10
        self.kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)

        # Network parameters
        self.epochs = 20
        self.verbose = 0

        # Network parameter search space
        # then we can go ahead and set the parameter space
        self.p = {'lr': (0.5, 5, 10),
             'first_neuron': [4, 8, 16, 32, 64],
             'hidden_layers': [0, 1, 2],
             'batch_size': (2, 30, 10),
             'epochs': [150],
             'dropout': (0, 0.5, 5),
             'weight_regulizer': [None],
             'emb_output_dims': [None],
             'shape': ['brick', 'long_funnel'],
             'optimizer': [Adam, Nadam, RMSprop],
             'losses': [logcosh, binary_crossentropy],
             'activation': [relu, elu],
             'last_activation': [sigmoid]}

        # Scores
        self.metric_loss = []
        self.metric_acc = []
        self.metric_dr = []
        self.metric_far = []

        self.metric_val_loss = []
        self.metric_val_acc = []
        self.metric_val_dr = []
        self.metric_val_far = []

        with timer('\nPreparing dataset'):
            self.load_data()
            self.set_y()
            self.remove_target_from_X()
            self.n_features = self.X.shape[1]
            self.train_test_split()

        with timer('\nTraining & validating model with kfold'):
            # Train model on K-1 and validate using remaining fold
            self.index = 0
            self.batch_size = False

            t = ta.Scan(x=self.X_train,
                        y=self.y_train,
                        model=self.get_model,
                        grid_downsample=0.01,
                        params=self.p,
                        dataset_name='kddcup',
                        experiment_no='1')

        # self.log_file()
        print('Finished')

    @staticmethod
    def dr(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = K.sum(y_pos * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)
        return tp / (tp + fn + K.epsilon())

    @staticmethod
    def far(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = K.sum(y_neg * y_pred_neg)
        fp = K.sum(y_neg * y_pred_pos)
        return fp / (tn + fp + K.epsilon())

    def get_model(self, x_train, y_train, x_val, y_val, params):

        model = models.Sequential()
        model.add(layers.Dense(self.n_features, activation=params['activation'], input_shape=(self.n_features,)))
        model.add(layers.Dropout(params['dropout']))
        model.add(layers.Dense(1, activation=params['last_activation'], kernel_initializer='normal'))
        model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), loss=params['losses'], metrics=['accuracy', self.dr, self.far])

        history = self.model.fit(x_train, y_train,
                                      validation_data=(x_val, y_val),
                                      batch_size=params['batch_size'],
                                      epochs=params['epochs'],
                                      verbose = 0)
        return history, model

    def log_file(self):
        if self.gettrace is None:
            pass
        elif self.gettrace():
            pass
        else:
            if self.logfile:
                sys.stdout = self.original_stdout
                self.logfile.close()
                self.logfile = False
            else:
                # Redirect stdout to file for logging if not in debug mode
                self.logfile = open('logs/{}_{}_stdout.txt'.format(self.__class__.__name__, self.timestr), 'w')
                sys.stdout = self.logfile

    def load_data(self):
        self.X = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_Tensor2d_type_1')
        print('\tRow count:\t', '{}'.format(self.X.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.X.shape[1]))

    def set_y(self):
        self.y = self.X['attack_category']
        self.y = self.y.map(self.label_map_string_2_int)

    def remove_target_from_X(self):
        self.X.drop('attack_category', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def map_target_to_label(self, t):
        return np.vectorize(self.label_map_int_2_string.get)(t)

    def fname(self, title):
        return '{}/{}.png'.format(self.folder, title)


annmlpbinary = AnnMLPBinary()

