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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize
import itertools


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
        self.splits = 3
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


class DecisionTreeClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.binary_enabled = False
        self.multi_enabled = False
        self.base['model'] = DecisionTreeClassifier(random_state=self.random_state)


class AnnPerceptronBinary(Model):
    def __init__(self, n_features):
        Model.__init__(self)
        self.binary_enabled = False
        self.multi_enabled = False
        self.epochs = 2
        self.batch_size = 100
        self.n_features = n_features
        self.base['model'] = self.create_network()

    def create_network(self):
        model = Sequential()
        model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.), input_shape=(self.n_features,)))
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train):
        self.base['model'].fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X_test):
        y_pred = self.base['model'].predict_classes(X_test)
        return y_pred.ravel()


class AnnPerceptronMulti(Model):
    def __init__(self, n_features):
        Model.__init__(self)
        self.binary_enabled = False
        self.multi_enabled = True
        self.epochs = 2
        self.batch_size = 100
        self.n_features = n_features
        self.base['model'] = self.create_network()

    def create_network(self):
        model = Sequential()
        model.add(Dense(units=5, activation='sigmoid', kernel_regularizer=l2(0.), input_shape=(self.n_features,)))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train):
        y_train = pd.get_dummies(y_train)  # for multi neural networks
        self.base['model'].fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

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
        self.n_features = None
        self.random_state = 20


        with timer('\nLoading dataset'):
            self.load_data()

        with timer('\nSetting X and y'):
            self.set_X()
            self.n_features = self.X.shape[1]

        models = (RandomForestClf(), DecisionTreeClf(), AnnPerceptronBinary(self.n_features),
                  AnnPerceptronMulti(self.n_features))
        classification_type = ('binary', 'multi')

        score = False
        for m, ctype in itertools.product(models, classification_type):
            if ctype == 'binary' and m.binary_enabled:
                self.set_y_binary()
                score = True
            elif ctype == 'multi' and m.multi_enabled:
                self.set_y_multi()
                score = True

            if not score:
                continue

            print('Processing {} {}'.format(m.__class__.__name__, ctype))

            m.score(self.X, self.y, ctype)
            yt = pd.Series(m.y_test[ctype])
            yp = pd.Series(m.y_pred[ctype])
            title = '{} - {} {} '.format('CM', m.__class__.__name__, ctype)
            self.visualize.confusion_matrix(yt, yp, title)

        # self.log_file()
        print('Finished')

    # look at loss=categorical_crossentropy for multiclass


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
        self.set_attack_category_to_binary()
        self.y = self.y.values.ravel()

    def set_y_multi(self):
        self.y = self.full.loc[:, ['attack_category']]
        self.set_attack_category_to_multi()
        self.y = self.y.values.ravel()

    def set_attack_category_to_binary(self):
        conditions = [
            (self.y['attack_category'] == 'normal'),
            (self.y['attack_category'] == 'dos') | (self.y['attack_category'] == 'u2r') |
            (self.y['attack_category'] == 'r2l') | (self.y['attack_category'] == 'probe')
        ]
        self.y['attack_category'] = np.select(conditions, [0, 1])

    def set_attack_category_to_multi(self):
#        conditions = [
#            (self.y['attack_category'] == 'normal'),
#            (self.y['attack_category'] == 'dos'), (self.y['attack_category'] == 'u2r'),
#            (self.y['attack_category'] == 'r2l'),  (self.y['attack_category'] == 'probe')
#        ]
#        self.y['attack_category'] = np.select(conditions, ['0', '1', '2', '3', '4'])
        pass


modelling = Modelling()

