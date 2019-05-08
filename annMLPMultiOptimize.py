"""
===========================================================================
Multi Layer Perceptron - Multiclass Optimize
===========================================================================
Multi Layer Perceptron - Multiclass Optimize
"""
import os
from contextlib import contextmanager
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import talos as ta
from talos.model.normalizers import lr_normalizer
import tensorflow as tf
import keras.backend as K
from keras import models, layers
from keras.optimizers import RMSprop, SGD
from keras.activations import relu, softmax
from filehandler import Filehandler
from dataset import KDDCup1999


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class AnnMLPMultiOptimize:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore low level instruction warnings
        tf.logging.set_verbosity(tf.logging.ERROR)  # Set tensorflow verbosity
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        print(__doc__)

        self.random_state = 20
        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.folder = 'tuning'

        # Datasets
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_features = None
        self.label_map_string_2_int = {'normal': 0, 'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4}

        with timer('\nPreparing dataset'):
            self.load_data()
            self.set_y()
            self.remove_target_from_X()
            self.n_features_all = self.X.shape[1]
            self.n_features_50pct = int(self.n_features_all * 0.5)
            self.n_features_80pct = int(self.n_features_all * 0.8)
            self.y = pd.get_dummies(self.y)
            self.X = self.X.values
            self.y = self.y.values

        with timer('\nSearching parameter space'):
            # self.p = {'lr': (0.5, 5, 10),
            #      'first_neuron': [self.n_features_70pct, self.n_features_all],
            #      'hidden_layers': [0, 1, 2],
            #      'hidden_neuron': [self.n_features_70pct, self.n_features_all],
            #      'batch_size': [100, 200],
            #      'epochs': [30],
            #      'dropout': (0, 0.2, 0.5),
            #      'weight_regulizer': [None],
            #      'emb_output_dims': [None],
            #      'shape': ['brick', 'long_funnel'],
            #      'optimizer': [Adam, RMSprop],
            #      'losses': [binary_crossentropy],
            #      'activation': [relu],
            #      'last_activation': [sigmoid]}

            self.ptest = {'lr': [10],
                       'first_neuron': [self.n_features_all],
                       'hidden_layers': [1],
                       'hidden_neuron': [self.n_features_all],
                       'batch_size': [100],
                       'epochs': [5],
                       'dropout': [0.2],
                       'optimizer': [SGD],
                       'activation': [relu],
                       'last_activation': [softmax]}

            self.p1 = {'lr': (0.5, 5, 10),
                       'first_neuron': [self.n_features_50pct, self.n_features_80pct, self.n_features_all],
                       'hidden_layers': [1, 2, 3],
                       'hidden_neuron': [self.n_features_50pct, self.n_features_80pct, self.n_features_all],
                       'batch_size': [100, 500, 1000],
                       'epochs': [20],
                       'dropout': (0, 0.2, 5),
                       'optimizer': [SGD, RMSprop],
                       'activation': [relu],
                       'last_activation': [softmax]}

            dataset_name = self.folder + '/Hyperparameter tuning - ' + self.__class__.__name__
            scan = ta.Scan(x=self.X,
                        y=self.y,
                        model=self.get_model,
                        params=self.p1,
                        grid_downsample=0.01,
                        dataset_name=dataset_name,
                        experiment_no='1')

            with timer('\nEvaluating Scan'):
                r = ta.Reporting(scan)

                # get the number of rounds in the Scan
                print('\nNumber of rounds in scan ', r.rounds())

                # get highest results
                print('\nHighest validation accuracy', r.high('val_dr'))
                print('\nHighest validation detection rate', r.high('val_dr'))
                print('\nHighest validation false alarm rate', r.high('val_far'))

                # get the highest result for any metric
                print(r.high('val_dr'))

                # get the round with the best result
                print('Best round', r.rounds2high())

                # get the best paramaters
                print(r.best_params())


                #r.plot_corr()
                #plt.show()

                # a four dimensional bar grid
                #r.plot_bars('batch_size', 'val_dr', 'hidden_layers', 'lr')
                #plt.show()

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

        # Input layer with dropout
        model.add(layers.Dense(params['first_neuron'], activation=params['activation'],
                               input_shape=(self.n_features_all,)))
        model.add(layers.Dropout(params['dropout']))

        # Hidden layers with dropout
        for i in range(params['hidden_layers']):
            model.add(layers.Dense(params['hidden_neuron'], activation=params['activation']))
            model.add(layers.Dropout(params['dropout']))

        # Output layer
        model.add(layers.Dense(5, activation=params['last_activation']))

        # Build model
        model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                      loss='categorical_crossentropy', metrics=['accuracy', self.dr, self.far])

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=params['batch_size'],
                            epochs=params['epochs'], verbose=0)

        return history, model

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


annmlpmulti = AnnMLPMultiOptimize()

