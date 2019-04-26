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
import matplotlib.pyplot as plt


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
        self.label_map_int_2_string = {0: 'normal', 1: 'dos', 2: 'u2r', 3: 'r2l', 4: 'probe'}
        self.label_map_string_2_int = {'normal': 0, 'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4}

        # K-fold validation
        self.splits = 5
        self.kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)

        # Network parameters
        self.epochs = 100
        self.batch_size = 50
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

            # Get single class prediction (rather than multi class probability summing to 1)
            y_pred = model.predict_classes(self.X_test)

            print('Test loss', np.mean(history.history['loss']))
            print('Test acc', np.mean(history.history['acc']))
            print('Accuracy {}'.format(accuracy_score(self.y_test, y_pred)))

            # Remap to string class targets
            y_pred = pd.Series(y_pred)
            y_pred = self.series_map_ac_multi_to_label(y_pred)
            self.y_test = self.series_map_ac_multi_to_label(self.y_test)

            # To numpy arrays for cm
            y_pred = y_pred.values
            self.y_test = self.y_test.values
            title = '{} - {} - {} '.format('CM', self.__class__.__name__, 'Multi')
            self.visualize.confusion_matrix(self.y_test, y_pred, title)

            epochs = range(1, len(history.history['loss']) + 1)

            plt.plot(epochs, np.mean(self.metric_loss, axis=0), 'bo', label='Training loss')
            plt.plot(epochs, np.mean(self.metric_val_loss, axis=0), 'b', label='Validation loss')
            plt.plot(epochs, history.history['loss'], 'b*', label='Test loss')
            plt.title('Training, validation and test loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.clf()
            plt.plot(epochs, np.mean(self.metric_acc, axis=0), 'bo', label='Training acc')
            plt.plot(epochs, np.mean(self.metric_val_acc, axis=0), 'b', label='Validation acc')
            plt.plot(epochs, history.history['acc'], 'b*', label='Test acc')
            plt.title('Training, validation and test acc')
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.legend()
            plt.show()

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
        self.y = self.y.map(self.label_map_string_2_int)
        #self.y = self.y.values.ravel()

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
        return s.map(self.label_map_int_2_string)


annmlpmulti = AnnMLPMulti()

