"""
==============================================================================
Feature selection techniques using KDD Cup 1999 IDS dataset
==============================================================================
For dataset with 5 attack categories in which classes are extremely imbalanced
"""
from contextlib import contextmanager
import time
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Using MinMax as SelectKBest does not support negs
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from filehandler import Filehandler
from visualize import Visualize
from dataset import KDDCup1999


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class FeatureSelector:
    def __init__(self):
        self.random_state = 20
        self.num_features = 15
        self.model = None
        self.fit = None
        self.title = ''
        self.title_suffix = ''
        self.fs_features_selected = None
        self.fs_n_features = None
        self.fs_feature_ranking = None
        self.fs_variance = None

    def show_title(self, label):
        self.title = self.__class__.__name__ + self.title_suffix + ' - Label ' + label
        print('\n--- ' + self.title)

    def fit_model(self, X, y):
        self.fit = self.model.fit(X, y)


class Original(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = SelectKBest(score_func=chi2, k=self.num_features)
        self.title_suffix = ' - All Features'

    def fit_model(self, X, y):
        pass

    def get_top_features(self, X, label):
        self.show_title(label)
        return X


class UnivariateSelector(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = SelectKBest(score_func=chi2, k=self.num_features)
        self.title_suffix = ' - With Chi-Squared'

    def get_top_features(self, X, label):
        self.show_title(label)
        cols = self.model.get_support()
        return X[X.columns[cols].tolist()]


class RecursiveSelector(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=600)
        self.title_suffix = ' - Recursive With Log. Regr.'
        self.rfe = RFE(self.model, self.num_features)

    def fit_model(self, X, y):
        self.fit = self.rfe.fit(X, y)

    def get_top_features(self, X, label):
        self.show_title(label)
        return X[X.columns[self.fit.support_].tolist()]


class PCASelector(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = PCA(n_components=self.num_features)
        self.title_suffix = ' - Principal Component Analysis'

    def fit_model(self, X, y):
        self.fit = self.model.fit_transform(X)

    def get_top_features(self, X, label):
        self.show_title(label)
        return self.fit


class KernelPCASelector(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = KernelPCA(n_components = self.num_features, kernel = 'rbf')
        self.title_suffix = ' - Kernel (RBF) PCA'

    def fit_model(self, X, y):
        self.fit = self.model.fit_transform(X)

    def get_top_features(self, X, label):
        self.show_title(label)
        return self.fit


class ExtraTreesSelector(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)
        self.title_suffix = ' - Extra Trees Clf'

    def get_top_features(self, X, label):
        self.show_title(label)
        feats = {}
        for feature, importance in zip(X.columns, self.model.feature_importances_):
            feats[feature] = importance

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
        importances.sort_values(by='importance', ascending=False, inplace=True)
        cols = importances.index.tolist()
        return X[cols[:self.num_features]]


class RandomForestSelector(FeatureSelector):
    def __init__(self):
        FeatureSelector.__init__(self)
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.title_suffix = ' - Random Forest Clf'

    def get_top_features(self, X, label):
        self.show_title(label)
        feats = {}
        for feature, importance in zip(X.columns, self.model.feature_importances_):
            feats[feature] = importance

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
        importances.sort_values(by='importance', ascending=False, inplace=True)
        cols = importances.index.tolist()
        return X[cols[:self.num_features]]


class FeatureSelection:
    def __init__(self):
        print(__doc__)
        self.filehandler = Filehandler()
        self.visualize = Visualize()
        self.ds = KDDCup1999()
        self.X = None
        self.y = None
        self.full = None
        self.random_state = 20
        self.num_features = 15
        self.scale_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'hot', 'num_failed_logins',
                           'logged_in', 'num_compromised', 'root_shell', 'num_file_creations', 'num_shells',
                           'num_access_files', 'count', 'srv_count', 'serror_rate', 'rerror_rate', 'diff_srv_rate',
                           'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_diff_srv_rate',
                           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate']

        with timer('\nLoading dataset'):
            self.load_data()
            self.encode_scale()
            self.set_X()
        with timer('\nFeature selection'):
            for selector in (Original(),
                             UnivariateSelector(),
                             RecursiveSelector(),
                             PCASelector(),
                             KernelPCASelector(),
                             ExtraTreesSelector(),
                             RandomForestSelector()):
                for label in ('attack_category', 'target'):
                    self.set_y(label)
                    selector.fit_model(self.X, self.y)
                    x = selector.get_top_features(self.X, label)
                    self.score_with_xgboost(x, self.y, selector.title)

    def load_data(self):
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')
        self.full = pd.concat([self.ds.dataset, self.ds.target], axis=1)
        self.ds.row_count_by_target('attack_category')

    def encode_scale(self):
        # Encode categoricals
        le = preprocessing.LabelEncoder()
        self.full['protocol_type'] = le.fit_transform(self.full['protocol_type'])
        self.full['service'] = le.fit_transform(self.full['service'])
        self.full['flag'] = le.fit_transform(self.full['flag'])

        # Scale
        sc = MinMaxScaler()
        self.full[self.scale_cols] = sc.fit_transform(self.full[self.scale_cols])

    def set_X(self):
        self.X = self.full.iloc[:, :-2]

    def set_y(self, label):
        self.y = self.full[label]

    def score_with_xgboost(self, x, y, title):
        clf = XGBClassifier(n_estimators=100, random_state=self.random_state)
        kfold = StratifiedKFold(n_splits=10, random_state=self.random_state)
        results = cross_val_score(clf, x, y, cv=kfold)
        print("XGBoost Accuracy: %.2f%% (+/- %.2f%%)" % (results.mean() * 100, results.std() * 100))
        y_pred = cross_val_predict(clf, x, y, cv=10)
        self.visualize.confusion_matrix(y, y_pred, title)


feature_selection = FeatureSelection()
