"""
===========================================================================
Logistic Regression
===========================================================================
Logistic Regression
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from filehandler import Filehandler
from dataset import KDDCup1999


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class LogisticRegressionBinary:
    def __init__(self):
        # self.logfile = None
        # self.gettrace = getattr(sys, 'gettrace', None)
        # self.original_stdout = sys.stdout
        # self.timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.log_file()

        print(__doc__)

        self.random_state = 20
        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.n_classes = 2

        # Datasets
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_map_int_2_string = {0: 'good', 1: 'bad', '0': 'good', '1': 'bad'}
        self.label_map_string_2_int = {'normal': 0, 'dos': 1, 'u2r': 1, 'r2l': 1, 'probe': 1}
        self.max_iters = 100

        with timer('\nPreparing dataset'):
            self.load_data()
            self.set_y()
            self.remove_target_from_X()
            self.train_test_split()

        with timer('\nTesting model on unseen test set'):
            lr = LogisticRegression(penalty='l2', solver='sag', max_iter=self.max_iters)
            lr.fit(self.X_train, self.y_train)
            self.y_pred = lr.predict(self.X_test)
            cm = confusion_matrix(self.y_test, self.y_pred)
            self.tp = self.get_tp_from_cm(cm)
            self.tn = self.get_tn_from_cm(cm)
            self.fp = self.get_fp_from_cm(cm)
            self.fn = self.get_fn_from_cm(cm)

            self.dr = self.tp / (self.tp + self.fp)
            self.far = self.fp / (self.tn + self.fp)
            self.acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
            print('Detection rate: ', self.dr)
            print('False alarm rate: ', self.far)
            print('Accuracy: ', self.acc)

        #self.log_file()
        print('Finished')




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


logisticregressionbinary = LogisticRegressionBinary()

