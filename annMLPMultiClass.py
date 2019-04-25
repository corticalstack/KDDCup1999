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
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from keras import models, layers
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


class AnnMLPMulti:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore low level instruction warnings
        tf.logging.set_verbosity(tf.logging.ERROR)  # Set tensorflow verbosity

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

        # Datasets
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_features = None
        self.label_multi = {0: 'normal', 1: 'dos', 2: 'u2r', 3: 'r2l', 4: 'probe'}

        # K-fold validation
        self.splits = 2
        self.kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)

        # Network parameters
        self.epochs = 10
        self.batch_size = 100
        self.verbose = 0

        # Scores
        self.metric_loss = []
        self.metric_acc = []
        self.metric_val_loss = []
        self.metric_val_acc = []

        with timer('\nPreparing dataset'):
            self.load_data()
            self.set_y()
            self.remove_target_from_X()
            self.n_features = self.X.shape[1]
            self.train_test_split()



            agg_ypred = []
            agg_ytest = []
        with timer('\nTraining & validating model with kfold'):
            # Train model on K-1 and validate using remaining fold
            index = 0
            for train, val in self.kfold.split(self.X_train, self.y_train):
                index += 1
                model = self.get_model()
                y_train_onehotencoded = pd.get_dummies(self.y_train[train])
                y_val_onehotencoded = pd.get_dummies(self.y_train[val])

                history = model.fit(self.X_train.iloc[train], y_train_onehotencoded,
                                    validation_data=(self.X_train.iloc[val], y_val_onehotencoded),
                                    epochs=self.epochs, batch_size=self.batch_size)
                self.metric_loss.append(history.history['loss'])
                self.metric_acc.append(history.history['acc'])
                self.metric_val_loss.append(history.history['val_loss'])
                self.metric_val_acc.append(history.history['val_acc'])

            print('Training mean loss', np.mean(self.metric_loss))
            print('Training mean acc', np.mean(self.metric_acc))
            print('Validation mean loss', np.mean(self.metric_val_loss))
            print('Validation mean acc', np.mean(self.metric_val_acc))

        with timer('\nTesting model on unseen test set'):
            model = self.get_model()
            y_test_onehotencoded = pd.get_dummies(self.y_test)
            y_train_onehotencoded = pd.get_dummies(self.y_train)

            # Train model on complete train set and validate with unseen test set
            history = model.fit(self.X_train, y_train_onehotencoded, validation_data=(self.X_test, y_test_onehotencoded),
                                epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
            print('Test loss', np.mean(history.history['loss']))
            print('Test acc', np.mean(history.history['acc']))

                #y_pred = self.predict(X.loc[test])
                #agg_ypred.append(y_pred)
                #agg_ytest.append(y[test])

        #self.y_pred[ctype] = [item for sublist in agg_ypred for item in sublist]
        #self.y_test[ctype] = [item for sublist in agg_ytest for item in sublist]

        #m.score(self.X_train, self.y_train, ctype)

        #m.y_test[ctype] = pd.Series(m.y_test[ctype])
        #m.y_pred[ctype] = pd.Series(m.y_pred[ctype])
        #m.y_test[ctype] = m.y_test[ctype].astype(int)
        #m.y_pred[ctype] = m.y_pred[ctype].astype(int)

        #if ctype == 'Binary':
        #    m.y_test[ctype] = self.series_map_ac_binary_to_label(m.y_test[ctype])
        #    m.y_pred[ctype] = self.series_map_ac_binary_to_label(m.y_pred[ctype])
        #else:
        #    m.y_test[ctype] = self.series_map_ac_multi_to_label(m.y_test[ctype])
        #    m.y_pred[ctype] = self.series_map_ac_multi_to_label(m.y_pred[ctype])

        #title = '{} - {} - {} '.format('CM', m.__class__.__name__, ctype)
        #self.visualize.confusion_matrix(m.y_test[ctype], m.y_pred[ctype], title)
        #self.scores(m.y_test[ctype], m.y_pred[ctype])

        # self.log_file()
        print('Finished')

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
        self.base['model'].fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                               callbacks=[tensorboard])

    def predict(self, X_test):
        y_pred = self.base['model'].predict_classes(X_test)
        return y_pred.ravel()

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

    def set_y(self):
        self.y = self.X['attack_category']
        self.y = self.y.values.ravel()

    def remove_target_from_X(self):
        self.X.drop('attack_category', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def df_map_ac_label_to_multi(self):
        conditions = [
           (self.y['attack_category'] == 'normal'),
           (self.y['attack_category'] == 'dos'), (self.y['attack_category'] == 'u2r'),
           (self.y['attack_category'] == 'r2l'),  (self.y['attack_category'] == 'probe')
        ]
        self.y['attack_category'] = np.select(conditions, ['0', '1', '2', '3', '4'])  # string for get_dummies encoding

    def series_map_ac_multi_to_label(self, s):
        return s.map(self.label_multi)

    def scores(self, y_test, y_pred):
        print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))
        print('F1 {}'.format(classification_report(y_test, y_pred, digits=10)))


annmlpmulti = AnnMLPMulti()

