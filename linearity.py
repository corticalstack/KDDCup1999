"""
============================================================================
Linear Data analysis with scatter graphs, convex hull and linear classifiers
============================================================================
"""
from contextlib import contextmanager
import time
import pandas as pd
from filehandler import Filehandler
from dataset import KDDCup1999
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from visualize import Visualize


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class LinearSeparability:
    def __init__(self):
        self.filehandler = Filehandler()
        self.visualize = Visualize()
        self.random_state = 20
        self.ds = KDDCup1999()
        self.X = None
        self.y = None
        self.sample = None
        self.full = None
        self.ac_count = {}
        self.scale_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'hot', 'num_failed_logins',
                           'logged_in', 'num_compromised', 'root_shell', 'num_file_creations', 'num_shells',
                           'num_access_files', 'count', 'srv_count', 'serror_rate', 'rerror_rate', 'diff_srv_rate',
                           'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_diff_srv_rate',
                           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate']
        self.full_weights = {'normal': 1, 'dos': 1, 'probe': 1, 'u2r': 1, 'r2l': 1}
        self.minimal_weights = {'normal': 0.01, 'dos': 0.01, 'probe': 0.2, 'u2r': 0.5, 'r2l': 0.5}

        with timer('\nLoading dataset'):
            self.load_data()
            self.set_attack_category_count()
            self.ds.shape()
        with timer('\nEncode and Scale dataset'):
            # Encode categoricals
            le = preprocessing.LabelEncoder()
            self.full['protocol_type'] = le.fit_transform(self.full['protocol_type'])
            self.full['service'] = le.fit_transform(self.full['service'])
            self.full['flag'] = le.fit_transform(self.full['flag'])

            # Scale
            sc = StandardScaler()
            self.full[self.scale_cols] = sc.fit_transform(self.full[self.scale_cols])
        with timer('\nPlotting scatter graphs'):
            self.sample_dataset(self.full_weights)
            print(self.sample.shape)
            self.scatter()
        with timer('\nPlotting scatter graphs with convex hull'):
            self.sample_dataset(self.full_weights)
            print(self.sample.shape)
            self.convex_hull()
        with timer('\nPlotting linear separability with perceptron'):
            self.sample_dataset(self.minimal_weights)
            print(self.sample.shape)
            self.classifiers()

    def scatter(self):
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='count', hue='attack_category')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='count', hue='target')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='serror_rate', hue='attack_category')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='dst_host_count', hue='attack_category')
        self.visualize.scatter(self.X, cola='rerror_rate', colb='count', hue='attack_category')
        self.visualize.scatter(self.X, cola='srv_diff_host_rate', colb='srv_count', hue='attack_category')

    def convex_hull(self):
        buckets = self.y.unique()
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='dst_host_srv_count', colb='count')
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='dst_host_srv_count', colb='serror_rate')
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='dst_host_srv_count', colb='dst_host_count')
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='rerror_rate', colb='count')

    def load_data(self):
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')
        self.full = pd.concat([self.ds.dataset, self.ds.target], axis=1)

    def set_attack_category_count(self):
        ac = self.full['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value

    def set_X_y(self):
        self.X = self.sample
        self.y = self.sample['attack_category']

    def sample_dataset(self, weights):
        self.sample = pd.DataFrame()
        for key, value in self.ac_count.items():
            samples = int(value * weights[key])
            df = self.full[self.full.attack_category == key].sample(samples, random_state=self.random_state)
            self.sample = self.sample.append(df)
        self.set_X_y()

    def classifiers(self):
        class_names = self.y.unique()
        le = preprocessing.LabelEncoder()
        self.y = le.fit_transform(self.y)
        _y = self.y

        models = (Perceptron(max_iter=100, tol=1e-3, random_state=self.random_state),
                  LinearSVC(max_iter=500, random_state=self.random_state, tol=1e-5),
                  SVC(kernel='rbf', gamma=1, C=1.0, random_state=self.random_state))

        titles = ('Perceptron', 'LinearSVC (linear kernel)', 'SVC with RBF kernel')
        columns = [('srv_diff_host_rate', 'srv_count'), ('dst_host_srv_count', 'count')]
        for clf, title in zip(models, titles):
            for cola, colb in columns:
                _x = self.X.loc[:, [cola, colb]]
                clf.fit(_x, _y)
                _y_pred = clf.predict(_x)
                self.visualize.boundary(_x, _y, clf, title, cola, colb)
                self.visualize.confusion_matrix(_y, _y_pred, title + ' - ' + cola + ' vs ' + colb, class_names)


linearSeparability = LinearSeparability()
