"""
====================================================
Clustering in 2D and 3D with and without PCA applied
====================================================
"""
from contextlib import contextmanager
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class Clustering:
    def __init__(self):
        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.visualize = Visualize()
        self.random_state = 20
        self.x = None
        self.y = None
        self.full = None
        self.n_clusters = 5
        self.colours = ['black', 'blue', 'red', 'cyan', 'green']
        self.ac_count = {}
        self.feature_idx = {0: 0, 1: 0, 2: 0}
        self.pca_idx = {0: 0, 1: 1, 2: 2, 'pca': True}
        self.scale_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'hot', 'num_failed_logins',
                           'logged_in', 'num_compromised', 'root_shell', 'num_file_creations', 'num_shells',
                           'num_access_files', 'count', 'srv_count', 'serror_rate', 'rerror_rate', 'diff_srv_rate',
                           'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_diff_srv_rate',
                           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate']
        self.cluster_cols = [('dst_host_srv_count', 'serror_rate', 'count'),
                             ('dst_host_srv_count', 'rerror_rate', 'serror_rate'),
                             ('srv_diff_host_rate', 'srv_count', 'serror_rate')]

        with timer('\nLoading dataset'):
            self.load_data()
            self.ds.shape()
            self.set_attack_category_count()
        with timer('\nEncode and Scale dataset'):
            self.encode_scale()
        with timer('\nSetting X and y'):
            self.set_x_y()
        with timer('\nPlotting clusters for specific columns'):
            for cola, colb, colc in self.cluster_cols:
                self.set_indexes(cola, colb, colc)
                with timer('\n2D clustering without PCA'):
                    self.cluster(idx=self.feature_idx)
                with timer('\n3D clustering without PCA'):
                    self.cluster(idx=self.feature_idx, projection='3d')
        with timer('\nPlotting clusters applying PCA'):
            with timer('\n2D clustering with PCA'):
                self.cluster(idx=self.pca_idx)
            with timer('\n3D clustering with PCA'):
                self.cluster(idx=self.pca_idx, projection='3d')

    def load_data(self):
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')
        self.full = pd.concat([self.ds.dataset, self.ds.target], axis=1)

    def set_attack_category_count(self):
        ac = self.full['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value

    def encode_scale(self):
        # Encode categoricals
        le = preprocessing.LabelEncoder()
        self.full['protocol_type'] = le.fit_transform(self.full['protocol_type'])
        self.full['service'] = le.fit_transform(self.full['service'])
        self.full['flag'] = le.fit_transform(self.full['flag'])

        # Scale
        sc = StandardScaler()
        self.full[self.scale_cols] = sc.fit_transform(self.full[self.scale_cols])

    def set_x_y(self):
        self.x = self.full.iloc[:, :-2]
        self.y = self.full['attack_category']

    def set_indexes(self, cola, colb, colc):
        self.feature_idx[0] = self.x.columns.get_loc(cola)
        self.feature_idx[1] = self.x.columns.get_loc(colb)
        self.feature_idx[2] = self.x.columns.get_loc(colc)

    def cluster(self, idx, projection=None):
        df_x = self.x
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(df_x)
        y_km = kmeans.fit_predict(df_x)
        self.visualize.scatter_clusters(self.x, self.n_clusters, y_km, idx, projection)


clustering = Clustering()
