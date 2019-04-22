"""
===========================================================================
Sampling techniques using KDD Cup 1999 IDS dataset
===========================================================================
The following examples demonstrate various sampling techniques for a dataset
in which classes are extremely imbalanced with heavily skewed features
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from collections import OrderedDict
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class Original:
    def fit_resample(self, x, y):
        return x, y


class Model:
    def __init__(self):
        self.enabled = False
        self.X_train = None
        self.y_train = None
        self.random_state = 20
        self.predictions = None
        self.base = {'model': None,
                     'stext': None,
                     'scores': None,
                     'cm': None}

    def fit(self, x, y):
        self.base['model'].fit(x, y)

    def predict(self, x, y):
        return cross_val_predict(self.base['model'], x, y, cv=10)


class XgboostClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.base['stext'] = 'XGC'
        self.base['model'] = XGBClassifier(n_estimators=100, random_state=self.random_state)


class Sampling:
    def __init__(self):
        self.logfile = None
        self.gettrace = getattr(sys, 'gettrace', None)
        self.original_stdout = sys.stdout
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.log_file()

        print(__doc__)

        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.visualize = Visualize()
        self.random_state = 20
        self.X = None
        self.y = None
        self.full = None

        # RF Feature selected plus sparse cols
        self.cols = ['count', 'diff_srv_rate', 'src_bytes', 'dst_host_srv_count', 'flag', 'dst_bytes', 'serror_rate',
                     'dst_host_diff_srv_rate', 'service', 'dst_host_count', 'dst_host_srv_diff_host_rate', 'logged_in',
                     'protocol_type', 'dst_host_same_src_port_rate', 'hot', 'srv_count', 'wrong_fragment',
                     'num_compromised', 'rerror_rate', 'srv_diff_host_rate', 'urgent', 'num_failed_logins',
                     'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                     'is_guest_login']

        with timer('\nLoading dataset'):
            self.load_data()

        with timer('\nScaling'):
            # Sampling options
            for sampler in (Original(),
                            RandomOverSampler(),
                            SMOTE(random_state=self.random_state),
                            ADASYN(random_state=self.random_state),
                            BorderlineSMOTE(random_state=self.random_state, kind='borderline-1')):

                self.X = self.full.loc[:, self.cols]
                self.X['target'] = self.full['target']
                print('X shape with selected features and binary - ', self.X.shape)

                self.X = pd.get_dummies(data=self.X, columns=['protocol_type', 'service', 'flag'])
                print('X shape after encoding categoricals - ', self.X.shape)

                # Re-sample based on attack_category labels
                res_x = pd.DataFrame()
                res_x, res_y_attack_category, title = self.sample(sampler, self.X, self.full['attack_category'])

                res_y_target = res_x['target']  # Grab target as y from resampled x set
                res_x.drop(columns=['target'], inplace=True)
                print('X shape after sampling and removing target - ', res_x.shape)
                print('y shape with attack_category after resample - ', res_y_attack_category.shape)
                print(res_y_attack_category.value_counts())
                res_y_attack_category.value_counts().plot(kind='bar', title=title + ' - Resampled Count (attack_category)')
                plt.show()
                print('y shape with target after resample - ', res_y_target.shape)

                # Scale after resampling
                qt = QuantileTransformer(output_distribution='normal')
                res_x = qt.fit_transform(res_x)
                print('X shape after scaling - ', res_x.shape)

                # Score on attack_category multi-class
                self.model_and_score(res_x, res_y_attack_category, title, 'attack_category')

                # Score on binary target
                self.model_and_score(res_x, res_y_target, title, 'target')

        self.log_file()
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
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')
        self.full = pd.concat([self.ds.dataset, self.ds.target], axis=1)
        self.ds.shape()
        self.ds.row_count_by_target('attack_category')

    def set_y(self, label):
        self.y = self.full[label]

    def sample(self, sampler, X, y):
        title = sampler.__class__.__name__
        res_x, res_y = sampler.fit_resample(X, y)
        if isinstance(res_x, np.ndarray):
            res_x = pd.DataFrame(res_x, columns=X.columns)

        if isinstance(res_y, np.ndarray):
            res_y = pd.Series(res_y)

        print('Shape after sampling with {} - x {},  y {}'.format(title, res_x.shape, res_y.shape))
        return res_x, res_y, title

    def model_and_score(self, X, y, title, label):
        clf = XGBClassifier(n_estimators=50, random_state=self.random_state)
        kfold = StratifiedKFold(n_splits=5, random_state=self.random_state)
        results = cross_val_score(clf, X, y, cv=kfold)
        y_pred = cross_val_predict(clf, X, y, cv=5)
        print('{} - {} - XGBoost Accuracy: {:.2f}% (+/- {:.2f}'.format(title, label, results.mean() * 100,
                                                                       results.std() * 100))
        self.visualize.confusion_matrix(y, y_pred, '{} - {} - Label {}'.format(title, clf.__class__.__name__, label))


sampling = Sampling()

