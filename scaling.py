"""
==========================================================
Scaling techniques using KDD Cup 1999 IDS dataset
==========================================================
The following examples demonstrate various scaling techniques
for a dataset in which classes are extremely imbalanced with heavily skewed features
"""
from contextlib import contextmanager
import time
import pandas as pd
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from collections import OrderedDict
import itertools
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


class RandomForestClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.base['stext'] = 'RFC'
        self.base['model'] = RandomForestClassifier(random_state=self.random_state)


class XgboostClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.base['stext'] = 'XGC'
        self.base['model'] = XGBClassifier(random_state=self.random_state)


class Scaling:
    def __init__(self):
        print(__doc__)
        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.visualize = Visualize()
        self.X = None
        self.y = None
        self.full = None
        self.class_names = None
        self.ac_count = {}
        self.scores = OrderedDict()
        self.scale_cols = ['duration', 'dst_host_srv_count', 'count', 'dst_host_count', 'rerror_rate', 'srv_diff_host_rate', 'srv_count']

        with timer('\nLoading dataset'):
            self.load_data()
            self.set_attack_category_count()
            self.set_X_y()
            self.ds.shape()
        with timer('\nScaling'):
            self.before_scaling()
            for scaler in (StandardScaler(),
                           Normalizer(),
                           MinMaxScaler(feature_range=(0, 1)),
                           Binarizer(threshold=0.0),
                           RobustScaler(quantile_range=(25, 75)),
                           PowerTransformer(method='yeo-johnson'),
                           QuantileTransformer(output_distribution='normal'),
                           QuantileTransformer(output_distribution='uniform')):
                self.scale(scaler)
        with timer('\nShowing Scores'):
            self.show_scores()

    def load_data(self):
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')
        self.full = pd.concat([self.ds.dataset, self.ds.target], axis=1)
        self.ds.row_count_by_target('attack_category')

    def set_attack_category_count(self):
        ac = self.full['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value

    def set_X_y(self):
        self.X = self.full.loc[:, self.scale_cols]
        self.y = self.full['attack_category']
        self.class_names = self.y.unique()

    def before_scaling(self):
        self.visualize.kdeplot('Before Scaling', self.X, self.scale_cols)

    def scale(self, scaler):
        x = self.X[self.scale_cols]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_x = scaler.fit_transform(x)

        res_x = pd.DataFrame(res_x, columns=self.scale_cols)

        title = 'After ' + scaler.__class__.__name__ + ' Scaler'
        self.visualize.kdeplot(title, res_x, self.scale_cols)

        for model in (RandomForestClf(),
                      XgboostClf()):
            model.fit(res_x, self.y)
            y_pred = model.predict(res_x, self.y)
            self.visualize.confusion_matrix(self.y, y_pred, title + ' - ' + model.__class__.__name__, self.class_names)
            self.register_score(scaler, model, res_x, self.y, y_pred)

    def register_score(self, scaler, clf, x, y, y_pred):
        prefix = scaler.__class__.__name__ + '_' + clf.__class__.__name__ + '_'

        # Warnings caught to suppress issues with minority classes having no predicted label values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + 'recall'] = recall_score(y, y_pred, average=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + 'precision'] = precision_score(y, y_pred, average=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + 'f1'] = f1_score(y, y_pred, average=None)

    def show_scores(self):
        print('--- Prediction Scores')
        for sid, score in self.scores.items():
            print('\nID: {}'.format(sid))
            print('\t\tScore{}'.format(score))


scaling = Scaling()
