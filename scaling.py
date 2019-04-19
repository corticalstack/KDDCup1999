"""
===========================================================================
Scaling techniques using KDD Cup 1999 IDS dataset
===========================================================================
The following examples demonstrate various scaling techniques for a dataset
in which classes are extremely imbalanced with heavily skewed features
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, RobustScaler, \
    QuantileTransformer, PowerTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from collections import OrderedDict
import warnings
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


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


class Scaling:
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
        self.X = None
        self.y = None
        self.full = None
        self.ac_count = {}
        self.scores = OrderedDict()
        self.scale_cols = ['count', 'diff_srv_rate', 'src_bytes', 'flag', 'dst_host_srv_count', 'dst_bytes',
                           'serror_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'logged_in',
                           'dst_host_count', 'hot', 'dst_host_srv_diff_host_rate', 'service', 'protocol_type',
                           'wrong_fragment', 'srv_count', 'num_compromised', 'rerror_rate', 'duration']

        with timer('\nLoading dataset'):
            self.load_data()
            self.set_attack_category_count()
        with timer('\nEncoding categoricals'):
            le = preprocessing.LabelEncoder()
            self.full['protocol_type'] = le.fit_transform(self.full['protocol_type'])
            self.full['service'] = le.fit_transform(self.full['service'])
            self.full['flag'] = le.fit_transform(self.full['flag'])
        with timer('\nSetting X'):
            self.set_X()
            self.ds.shape()
        with timer('\nDistribution Before Scaling'):
            self.dist_before_scaling()
        with timer('\nScaling'):
            for scaler in (StandardScaler(),
                           Normalizer(),
                           MinMaxScaler(feature_range=(0, 1)),
                           Binarizer(threshold=0.0),
                           RobustScaler(quantile_range=(25, 75)),
                           PowerTransformer(method='yeo-johnson'),
                           QuantileTransformer(output_distribution='normal'),
                           QuantileTransformer(output_distribution='uniform')):
                title, res_x = self.scale(scaler)

                label = 'attack_category'
                self.set_y(label)
                self.model_and_score(scaler, res_x, title, label)

                label = 'target'
                self.set_y(label)
                self.model_and_score(scaler, res_x, title, label)

        with timer('\nShowing Scores'):
            self.show_scores()

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

    def set_attack_category_count(self):
        ac = self.full['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value

    def set_X(self):
        self.X = self.full.loc[:, self.scale_cols]

    def set_y(self, label):
        self.y = self.full[label]

    def dist_before_scaling(self):
        self.visualize.kdeplot('Distribution Before Scaling', self.X, self.scale_cols)

    def scale(self, scaler):
        x = self.X[self.scale_cols]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_x = scaler.fit_transform(x)

        res_x = pd.DataFrame(res_x, columns=self.scale_cols)
        title = 'Distribution After ' + scaler.__class__.__name__
        self.visualize.kdeplot(title, res_x, self.scale_cols)
        return title, res_x

    def model_and_score(self, scaler, res_x, title, label):
        model = XgboostClf()
        model.fit(res_x, self.y)
        y_pred = model.predict(res_x, self.y)
        self.visualize.confusion_matrix(self.y, y_pred, '{} - {} - Label {}'.format(title, model.__class__.__name__,
                                                                                    label))
        self.register_score(scaler, model, res_x, self.y, y_pred, label)

    def register_score(self, scaler, clf, x, y, y_pred, label):
        prefix = '{}_{} - Label {}'.format(scaler.__class__.__name__, clf.__class__.__name__, label)

        # Warnings caught to suppress issues with minority classes having no predicted label values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + ' - recall'] = recall_score(y, y_pred, average=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + ' - precision'] = precision_score(y, y_pred, average=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + ' - f1'] = f1_score(y, y_pred, average=None)

    def show_scores(self):
        print('--- Prediction Scores')
        for sid, score in self.scores.items():
            print('\nID: {}'.format(sid))
            print('\t\tScore{}'.format(score))


scaling = Scaling()
