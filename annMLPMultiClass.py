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
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import models, layers
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize
import itertools
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import keras.backend as K


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
        self.n_classes = 5
        self.label_map_int_2_string = {0: 'normal', 1: 'dos', 2: 'u2r', 3: 'r2l', 4: 'probe'}
        self.label_map_string_2_int = {'normal': 0, 'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4}

        # K-fold validation
        self.splits = 2
        self.kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)

        # Network parameters
        self.epochs = 3
        self.batch_size = 100 #    # batch_size is not a divisor of the training set size
        self.verbose = 0

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
            for train, val in self.kfold.split(self.X_train, self.y_train):
                self.index += 1
                self.tensorboard = TensorBoard(log_dir='logs/tb/annmlpmulticlass_cv{}_{}'.format(self.index, time))
                self.model = self.get_model()
                self.y_train_onehotencoded = pd.get_dummies(self.y_train.iloc[train])
                self.y_val_onehotencoded = pd.get_dummies(self.y_train.iloc[val])

                self.history = self.model.fit(self.X_train.iloc[train], self.y_train_onehotencoded,
                                              validation_data=(self.X_train.iloc[val], self.y_val_onehotencoded),
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

            print('Training mean loss', np.mean(self.metric_loss))
            print('Training mean acc', np.mean(self.metric_acc))
            print('Training mean dr', np.mean(self.metric_dr))
            print('Training mean far', np.mean(self.metric_far))
            print('Validation mean loss', np.mean(self.metric_val_loss))
            print('Validation mean acc', np.mean(self.metric_val_acc))
            print('Validation mean dr', np.mean(self.metric_val_dr))
            print('Validation mean far', np.mean(self.metric_val_far))

        with timer('\nTesting model on unseen test set'):
            self.tensorboard = TensorBoard(log_dir='logs/tb/annmlpmulticlass_test_{}'.format(time))
            self.model = self.get_model()
            self.y_test_onehotencoded = pd.get_dummies(self.y_test)
            self.y_train_onehotencoded = pd.get_dummies(self.y_train)

            # Train model on complete train set and validate with unseen test set
            self.history = self.model.fit(self.X_train, self.y_train_onehotencoded,
                                          validation_data=(self.X_test, self.y_test_onehotencoded),
                                          epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                                          callbacks=[self.tensorboard])

            # Get single class prediction (rather than multi class probability summing to 1)
            y_pred = self.model.predict_classes(self.X_test)

            print('Test loss', np.mean(self.history.history['loss']))
            print('Test acc', np.mean(self.history.history['acc']))
            print('Test dr', np.mean(self.history.history['dr']))
            print('Test far', np.mean(self.history.history['far']))
            print('Accuracy {}'.format(accuracy_score(self.y_test, y_pred)))

            # Remap to string class targets
            self.y_pred = pd.Series(y_pred)
            self.y_pred = self.series_map_ac_multi_to_label(self.y_pred)
            self.y_test = self.series_map_ac_multi_to_label(self.y_test)

            # To numpy arrays for cm
            self.y_pred = self.y_pred.values
            self.y_test = self.y_test.values
            self.title = '{} - {} - {} '.format('CM', self.__class__.__name__, 'Multi')
            self.visualize.confusion_matrix(self.y_test, self.y_pred, self.title)

            # print('total from cm ', tp + tn + fp + fn, ' Size of test', self.y_test.shape)
            cm = confusion_matrix(self.y_test, self.y_pred)
            self.get_tp_from_cm(cm)
            self.get_tn_from_cm(cm)
            self.get_fp_from_cm(cm)
            self.get_fn_from_cm(cm)

            epochs = range(1, len(self.history.history['loss']) + 1)

            # Plot loss
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_loss, axis=0), 'g', label='Training loss')
            ax.plot(epochs, np.mean(self.metric_val_loss, axis=0), 'b', label='Validation loss')
            ax.plot(epochs, self.history.history['loss'], 'r', label='Test loss')
            plt.title('Training, validation and test loss', fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.legend()
            plt.show()

            # Plot accuracy
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_acc, axis=0), 'g', label='Training accuracy')
            ax.plot(epochs, np.mean(self.metric_val_acc, axis=0), 'b', label='Validation accuracy')
            ax.plot(epochs, self.history.history['acc'], 'r', label='Test accuracy')
            plt.title('Training, validation and test accuracy', fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.legend()
            plt.show()

            # Plot detection rate
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_dr, axis=0), 'g', label='Training detection rate')
            ax.plot(epochs, np.mean(self.metric_val_dr, axis=0), 'b', label='Validation detection rate')
            ax.plot(epochs, self.history.history['dr'], 'r', label='Test detection rate')
            plt.title('Training, validation and test detection rate', fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Detection Rate', fontsize=14)
            plt.legend()
            plt.show()

            # Plot false alarm rate
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))
            plt.style.use('ggplot')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.plot(epochs, np.mean(self.metric_far, axis=0), 'g', label='Training false alarm rate')
            ax.plot(epochs, np.mean(self.metric_val_far, axis=0), 'b', label='Validation false alarm rate')
            ax.plot(epochs, self.history.history['far'], 'r', label='Test false alarm rate')
            plt.title('Training, validation and test false alarm rate', fontsize=18)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('False Alarm Rate', fontsize=14)
            plt.legend()
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
        model.add(layers.Dense(self.n_features, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', self.dr, self.far])
        return model

    # True positives are the diagonal elements
    def get_tp_from_cm(self, cm):
        tp = np.diag(cm)
        print('tp', np.sum(np.diag(cm)))
        return np.sum(tp)

    def get_tn_from_cm(self, cm):
        tn = []
        for i in range(self.n_classes):
            temp = np.delete(cm, i, 0)  # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            tn.append(sum(sum(temp)))
        print('tn ', np.sum(tn))
        return np.sum(tn)

    # Sum of columns minus diagonal
    def get_fp_from_cm(self, cm):
        fp = []
        for i in range(self.n_classes):
            fp.append(sum(cm[:, i]) - cm[i, i])
        print('fp ', np.sum(fp))
        return np.sum(fp)

    # Sum of rows minus diagonal
    def get_fn_from_cm(self, cm):
        fn = []
        for i in range(self.n_classes):
            fn.append(sum(cm[i, :]) - cm[i, i])
        print('fn', np.sum(fn))
        return np.sum(fn)

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


    def remove_target_from_X(self):
        self.X.drop('attack_category', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def series_map_ac_multi_to_label(self, s):
        return s.map(self.label_map_int_2_string)


annmlpmulti = AnnMLPMulti()

