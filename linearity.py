"""
============================================================================
Linear Data analysis with scatter graphs, convex hull and linear classifiers
============================================================================
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class Linearity:
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
        self.sample = None
        self.full = None
        self.ac_count = {}
        self.scale_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                           'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                           'num_access_files', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'rerror_rate', 'diff_srv_rate',
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
            self.set_X_y('target')
            self.scatter()
        with timer('\nPlotting scatter graphs with convex hull'):
            self.sample_dataset(self.full_weights)
            print(self.sample.shape)
            self.set_X_y('target')
            self.convex_hull()
        with timer('\nPlotting linear separability with classifiers'):
            self.sample_dataset(self.minimal_weights)
            print(self.sample.shape)
            self.set_X_y('target')
            self.classifiers()

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

    def scatter(self):
        self.visualize.scatter(self.X, cola='src_bytes', colb='dst_bytes', hue='target')
        self.visualize.scatter(self.X, cola='count', colb='diff_srv_rate', hue='target')
        self.visualize.scatter(self.X, cola='duration', colb='src_bytes', hue='target')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='dst_bytes', hue='target')
        self.visualize.scatter(self.X, cola='serror_rate', colb='rerror_rate', hue='target')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='dst_bytes', hue='target')
        self.visualize.scatter(self.X, cola='srv_diff_host_rate', colb='srv_count', hue='target')

    def convex_hull(self):
        buckets = self.y.unique()
        self.visualize.convex_hull(self.X, buckets, cola='src_bytes', colb='dst_bytes', target='target')
        self.visualize.convex_hull(self.X, buckets, cola='count', colb='diff_srv_rate', target='target')
        self.visualize.convex_hull(self.X, buckets, cola='duration', colb='src_bytes', target='target')
        self.visualize.convex_hull(self.X, buckets, cola='dst_host_srv_count', colb='dst_bytes', target='target')
        self.visualize.convex_hull(self.X, buckets, cola='serror_rate', colb='rerror_rate', target='target')
        self.visualize.convex_hull(self.X, buckets, cola='dst_host_srv_count', colb='dst_bytes', target='target')
        self.visualize.convex_hull(self.X, buckets, cola='srv_diff_host_rate', colb='srv_count', target='target')

    def load_data(self):
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')
        self.full = pd.concat([self.ds.dataset, self.ds.target], axis=1)

    def set_attack_category_count(self):
        ac = self.full['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value

    def set_X_y(self, target):
        print('Setting X, with y as {}'.format(target))
        self.X = self.sample
        self.y = self.sample[target]

    def sample_dataset(self, weights):
        print('Sampling dataset with weights {}'.format(weights))
        self.sample = pd.DataFrame()
        for key, value in self.ac_count.items():
            samples = int(value * weights[key])
            df = self.full[self.full.attack_category == key].sample(samples, random_state=self.random_state)
            self.sample = self.sample.append(df)

    def classifiers(self):
        le = preprocessing.LabelEncoder()
        self.y = le.fit_transform(self.y)
        _y = self.y

        models = (Perceptron(max_iter=100, tol=1e-3, random_state=self.random_state),
                  LinearSVC(max_iter=500, random_state=self.random_state, tol=1e-5),
                  SVC(kernel='rbf', gamma=5, C=10.0, random_state=self.random_state))

        titles = ('Perceptron', 'LinearSVC (linear kernel)', 'SVC with RBF kernel')
        columns = [('srv_diff_host_rate', 'srv_count'), ('dst_host_srv_count', 'count'),
                   ('dst_host_srv_count', 'dst_bytes')]
        for clf, title in zip(models, titles):
            for cola, colb in columns:
                _x = self.X.loc[:, [cola, colb]]
                clf.fit(_x, _y)
                _y_pred = clf.predict(_x)
                self.visualize.boundary(_x, _y, clf, title, cola, colb)
                self.visualize.confusion_matrix(_y, _y_pred, title + ' - ' + cola + ' vs ' + colb)


linearity = Linearity()

