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
import keras.backend as K
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



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
            for train, val in self.kfold.split(self.X_train, self.y_train):
                if not self.batch_size:
                    self.batch_size = len(train) // 1000
                    print('Batch size set at {}'.format(self.batch_size))
                self.index += 1
                self.tensorboard = TensorBoard(log_dir='logs/tb/annmlpmulticlass_cv{}_{}'.format(self.index, time))
                self.model = self.get_model()

                self.history = self.model.fit(self.X_train.iloc[train], self.y_train.iloc[train],
                                              validation_data=(self.X_train.iloc[val], self.y_train.iloc[val]),
                                              epochs=self.epochs, batch_size=self.batch_size,
                                              callbacks=[self.tensorboard])

                self.metric_loss.append(self.history.history['loss'])
                self.metric_acc.append(self.history.history['acc'])
                self.metric_dr.append(self.history.history['dr'])
                self.metric_far.append(self.history.history['far'])
                self.metric_val_loss.append(self.history.history['val_loss'])
                self.metric_val_acc.append(self.history.history['val_acc'])
                self.metric_val_dr.append(self.history.history['val_dr'])
                self.metric_val_far.append(self.history.history['val_far'])

            print('\nTraining mean loss', np.mean(self.metric_loss))
            print('Training mean acc', np.mean(self.metric_acc))
            print('Training mean dr', np.mean(self.metric_dr))
            print('Training mean far', np.mean(self.metric_far))
            print('\nValidation mean loss', np.mean(self.metric_val_loss))
            print('Validation mean acc', np.mean(self.metric_val_acc))
            print('Validation mean dr', np.mean(self.metric_val_dr))
            print('Validation mean far', np.mean(self.metric_val_far))

        with timer('\nTesting model on unseen test set'):
            self.tensorboard = TensorBoard(log_dir='logs/tb/annmlpmulticlass_test_{}'.format(time))
            tf.reset_default_graph()  # Reset graph for tensorboard display

            K.clear_session()
            self.model = self.get_model()

            # Train model on complete train set and validate with unseen test set
            self.history = self.model.fit(self.X_train, self.y_train,
                                          validation_data=(self.X_test, self.y_test),
                                          epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                                          callbacks=[self.tensorboard])

            # Get single class prediction (rather than multi class probability summing to 1)
            y_pred = self.model.predict_classes(self.X_test)

            print('Test loss', np.mean(self.history.history['loss']))
            print('Test acc', np.mean(self.history.history['acc']))
            print('Test dr', np.mean(self.history.history['dr']))
            print('Test far', np.mean(self.history.history['far']))

            # Remap to string class targets
            self.y_pred = self.map_target_to_label(y_pred)
            self.y_pred = self.y_pred.ravel()
            self.y_test = self.map_target_to_label(self.y_test)

            self.visualize.confusion_matrix(self.y_test, self.y_pred, self.__class__.__name__)

            epochs = range(1, len(self.history.history['loss']) + 1)

            # Plot loss
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_loss, axis=0), 'g', label='Training')
            ax.plot(epochs, np.mean(self.metric_val_loss, axis=0), 'b', label='Validation')
            ax.plot(epochs, self.history.history['loss'], 'r', label='Test')
            self.title = '{} - {}'.format(self.__class__.__name__, 'Loss')
            plt.title(self.title, fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.legend(loc=1, prop={'size': 14})
            plt.savefig(fname=self.fname(self.title), dpi=300, format='png')
            plt.show()

            # Plot accuracy
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_acc, axis=0), 'g', label='Training')
            ax.plot(epochs, np.mean(self.metric_val_acc, axis=0), 'b', label='Validation')
            ax.plot(epochs, self.history.history['acc'], 'r', label='Test')
            self.title = '{} - {}'.format(self.__class__.__name__, 'Accuracy')
            plt.title(self.title, fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.legend(loc=4, prop={'size': 14})
            plt.savefig(fname=self.fname(self.title), dpi=300, format='png')
            plt.show()

            # Plot detection rate
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_dr, axis=0), 'g', label='Training')
            ax.plot(epochs, np.mean(self.metric_val_dr, axis=0), 'b', label='Validation')
            ax.plot(epochs, self.history.history['dr'], 'r', label='Test')
            self.title = '{} - {}'.format(self.__class__.__name__, 'Detection Rate')
            plt.title(self.title, fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Detection Rate', fontsize=14)
            plt.legend(loc=4, prop={'size': 14})
            plt.savefig(fname=self.fname(self.title), dpi=300, format='png')
            plt.show()

            # Plot false alarm rate
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_far, axis=0), 'g', label='Training')
            ax.plot(epochs, np.mean(self.metric_val_far, axis=0), 'b', label='Validation')
            ax.plot(epochs, self.history.history['far'], 'r', label='Test')
            self.title = '{} - {}'.format(self.__class__.__name__, 'False Alarm Rate')
            plt.title(self.title, fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('False Alarm Rate', fontsize=14)
            plt.legend(loc=1, prop={'size': 14})
            plt.savefig(fname=self.fname(self.title), dpi=300, format='png')
            plt.show()

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

    def get_model(self):
        model = models.Sequential()
        model.add(layers.Dense(self.n_features, activation='relu', input_shape=(self.n_features,)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.n_features, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', self.dr, self.far])
        return model

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

