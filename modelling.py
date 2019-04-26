"""
===========================================================================
Sampling techniques using KDD Cup 1999 IDS dataset
===========================================================================
The following examples demonstrate various sampling techniques for a dataset
in which classes are extremely imbalanced with heavily skewed features
"""
import os
import sys
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.model_selection import cross_val_predict, cross_val_score
import tensorflow as tf
from keras import models, layers
from keras.regularizers import l2
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize
import itertools
from tensorflow.python.keras.callbacks import TensorBoard


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class Model:

    def __init__(self):
        self.random_state = 20
        self.base = {'model': None}
        self.binary_enabled = False
        self.multi_enabled = False
        self.X = None
        self.y = None
        self.y_pred = {'binary': [], 'multi': []}
        self.y_test = {'binary': [], 'multi': []}
        self.splits = 2
        self.kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)

    def fit(self, X_train, y_train):
        self.base['model'].fit(X_train, y_train)

    def predict(self, X_test):
        return self.base['model'].predict(X_test)

    def score(self, X, y, ctype):
        agg_ypred = []
        agg_ytest = []
        for train, test in self.kfold.split(X, y):
            self.fit(X.loc[train], y[train])
            y_pred = self.predict(X.loc[test])
            agg_ypred.append(y_pred)
            agg_ytest.append(y[test])

        self.y_pred[ctype] = [item for sublist in agg_ypred for item in sublist]
        self.y_test[ctype] = [item for sublist in agg_ytest for item in sublist]


class RandomForestClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.binary_enabled = False
        self.multi_enabled = False
        self.base['model'] = RandomForestClassifier(n_estimators=100, random_state=self.random_state)


# Single Layer Perceptron - Binary Classification
class AnnSLPBinary(Model):
    def __init__(self, n_features):
        Model.__init__(self)
        self.binary_enabled = False
        self.epochs = 2
        self.batch_size = 100
        self.verbose = 0
        self.n_features = n_features
        self.base['model'] = self.create_network()

    def create_network(self):
        model = models.Sequential()
        model.add(layers.Dense(1, activation='sigmoid', input_shape=(self.n_features,)))
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train):
        self.base['model'].fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X_test):
        y_pred = self.base['model'].predict_classes(X_test)
        return y_pred.ravel()


# Multi Layer Perceptron - Binary Classification
class AnnMLPBinary(Model):
    def __init__(self, n_features):
        Model.__init__(self)
        self.binary_enabled = False
        self.epochs = 2
        self.batch_size = 100
        self.verbose = 0
        self.n_features = n_features
        self.base['model'] = self.create_network()

    def create_network(self):
        model = models.Sequential()
        model.add(layers.Dense(self.n_features, activation='relu', input_shape=(self.n_features,)))
        model.add(layers.Dense(self.n_features, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train):
        self.base['model'].fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X_test):
        y_pred = self.base['model'].predict_classes(X_test)
        return y_pred.ravel()


class AnnMLPMulti(Model):
    def __init__(self, n_features):
        Model.__init__(self)
        self.multi_enabled = True
        self.epochs = 20
        self.batch_size = 100
        self.verbose = 0
        self.n_features = n_features
        self.base = {}

    def get_model(self):
        model = models.Sequential()
        model.add(layers.Dense(self.n_features, activation='relu', input_shape=(self.n_features,)))
        model.add(layers.Dense(self.n_features, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
        tensorboard = TensorBoard(log_dir='logs/{}'.format(time))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train):
        y_train = pd.get_dummies(y_train)  # for multi neural networks
        tensorboard = TensorBoard(log_dir='logs/tensorboard/{}'.format(time.strftime("%Y%m%d-%H%M%S")))
        self.base['model'].fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                               callbacks=[tensorboard])

    def predict(self, X_test):
        y_pred = self.base['model'].predict_classes(X_test)
        return y_pred.ravel()


class Modelling:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore low level instruction warnings
        tf.logging.set_verbosity(tf.logging.ERROR)  # Set tensorflow verbosity

        # self.logfile = None
        # self.gettrace = getattr(sys, 'gettrace', None)
        # self.original_stdout = sys.stdout
        # self.timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.log_file()

        print(__doc__)

        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.visualize = Visualize()
        self.full = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_features = None
        self.random_state = 20
        self.label_multi = {0: 'normal', '0': 'normal', 1: 'dos', '1': 'dos', 2: 'u2r', '2': 'u2r', 3: 'r2l',
                            '3': 'r2l', 4: 'probe', '4': 'probe'}
        self.label_binary = {0: 'good', '0': 'good', 1: 'bad', '1': 'bad'}

        with timer('\nLoading dataset'):
            self.load_data()

        with timer('\nSetting X and y'):
            self.set_X()
            self.n_features = self.X.shape[1]

        models = (RandomForestClf(), AnnSLPBinary(self.n_features), AnnMLPBinary(self.n_features),
                  AnnMLPMulti(self.n_features))
        classification_type = ('Binary', 'Multi')

        for m, ctype in itertools.product(models, classification_type):
            score = False
            if ctype == 'Binary' and m.binary_enabled:
                self.set_y_binary()
                score = True
            elif ctype == 'Multi' and m.multi_enabled:
                self.set_y_multi()
                score = True

            if not score:
                continue

            with timer('\nTraining and scoring {} - {} target'.format(m.__class__.__name__, ctype)):
                m.base['model'] = m.get_model()
                #self.train_test_split()
                m.score(self.X, self.y, ctype)

            m.y_test[ctype] = pd.Series(m.y_test[ctype])
            m.y_pred[ctype] = pd.Series(m.y_pred[ctype])
            m.y_test[ctype] = m.y_test[ctype].astype(int)
            m.y_pred[ctype] = m.y_pred[ctype].astype(int)


            if ctype == 'Binary':
                m.y_test[ctype] = self.series_map_ac_binary_to_label(m.y_test[ctype])
                m.y_pred[ctype] = self.series_map_ac_binary_to_label(m.y_pred[ctype])
            else:
                m.y_test[ctype] = self.series_map_ac_multi_to_label(m.y_test[ctype])
                m.y_pred[ctype] = self.series_map_ac_multi_to_label(m.y_pred[ctype])


            title = '{} - {} - {} '.format('CM', m.__class__.__name__, ctype)
            self.visualize.confusion_matrix(m.y_test[ctype], m.y_pred[ctype], title)
            self.scores(m.y_test[ctype], m.y_pred[ctype])

    # Append the scores to a scores array. I could then do an np.mean(scores) to get the mean(average) from all the kfolds
    # save the epoch number and gfold number if possible as well, to get a per/epoch score

        # self.log_file()
        print('Finished')

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
        self.full = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_Tensor2d_type_1')

    def set_X(self):
        self.X = self.full.loc[:, self.full.columns != 'attack_category']

    def set_y_binary(self):
        self.y = self.full.loc[:, ['attack_category']]
        self.df_map_ac_label_to_binary()
        self.y = self.y.values.ravel()

    def set_y_multi(self):
        self.y = self.full.loc[:, ['attack_category']]
        self.df_map_ac_label_to_multi()
        self.y = self.y.values.ravel()

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def df_map_ac_label_to_binary(self):
        conditions = [
            (self.y['attack_category'] == 'normal'),
            (self.y['attack_category'] == 'dos') | (self.y['attack_category'] == 'u2r') |
            (self.y['attack_category'] == 'r2l') | (self.y['attack_category'] == 'probe')
        ]
        self.y['attack_category'] = np.select(conditions, [0, 1])

    def df_map_ac_label_to_multi(self):
        conditions = [
           (self.y['attack_category'] == 'normal'),
           (self.y['attack_category'] == 'dos'), (self.y['attack_category'] == 'u2r'),
           (self.y['attack_category'] == 'r2l'),  (self.y['attack_category'] == 'probe')
        ]
        self.y['attack_category'] = np.select(conditions, ['0', '1', '2', '3', '4'])  # string for get_dummies encoding

    def series_map_ac_multi_to_label(self, s):
        return s.map(self.label_multi)

    def series_map_ac_binary_to_label(self, s):
        return s.map(self.label_binary)

    def scores(self, y_test, y_pred):
        print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))
        print('F1 {}'.format(classification_report(y_test, y_pred, digits=10)))


modelling = Modelling()

# class AnnFeedForward(Model):
#
#    # Because this is a binary classification problem, one common choice is to use the sigmoid activation function in a one-unit output layer.
#
#     # Start neural network
#     network = models.Sequential()
#
#     # Add fully connected layer with a ReLU activation function
#     network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))
#
#     # Add fully connected layer with a ReLU activation function
#     network.add(layers.Dense(units=16, activation='relu'))
#
#     # Add fully connected layer with a sigmoid activation function
#     network.add(layers.Dense(units=1, activation='sigmoid'))
#
#     # Compile neural network
#     network.compile(loss='binary_crossentropy', # Cross-entropy
#                     optimizer='rmsprop', # Root Mean Square Propagation
#                     metrics=['accuracy']) # Accuracy performance metric
#
#
#     # Train neural network
#     history = network.fit(train_features,  # Features
#                           train_target,  # Target vector
#                           epochs=3,  # Number of epochs
#                           verbose=1,  # Print description after each epoch
#                           batch_size=100,  # Number of observations per batch
#                           validation_data=(test_features, test_target))  # Data for evaluation
#
# class ANNPerceptronClf(Model):
#     def __init__(self):
#         Model.__init__(self)
#         self.enabled = False
#         self.base['stext'] = 'ANNPCLF'
#         self.base['model'] = KerasClassifier(build_fn=self.create_network, epochs=10, batch_size=100, verbose=0)
#
#     def create_network(self):
#         network = Sequential()
#
#         # Input layer with inputs matching 0 axis of tensor, hidden layer with 1 neuron
#         network.add(Dense(output_dim=1, init='uniform', activation='relu', input_dim=self.X_train.shape[1]))
#
#         # Output layer - sigmoid good for binary classification
#         network.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
#
#         # Binary cross entropy good for binary classification
#         network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#         return network
#
#     def set_dataset(self, folder, file):
#         Model.set_dataset(self, folder, file)
#
#     def fit(self):
#         self.base['model'].fit(self.X_train, self.y_train)
#
#     def predict(self):
#         self.predictions = self.base['model'].predict(self.X_train)
#         self.predictions = (self.predictions > 0.5)

# class AnnMlpBinary(Model):
#     def __init__(self):
#         Model.__init__(self)
#         self.enabled = False
#         self.base['stext'] = 'ANNPCLF'
#         self.base['model'] = KerasClassifier(build_fn=self.create_network, epochs=10, batch_size=100, verbose=0)
#
#     def create_network(self):
#         model = Sequential()
#         model.add(Dense(64, input_dim=20, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(1, activation='sigmoid'))
#
#         model.compile(loss='binary_crossentropy',
#                       optimizer='rmsprop',
#                       metrics=['accuracy'])
#
#         model.fit(x_train, y_train,
#                   epochs=20,
#                   batch_size=128)
#         score = model.evaluate(x_test, y_test, batch_size=128)
