"""
===========================================================================
Prepare tensor input files for neural network modelling
===========================================================================
The following prepares 2D tensors as input to neural networks
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE
from filehandler import Filehandler
from dataset import KDDCup1999
from sklearn.decomposition import PCA


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class Tensor2d:
    def __init__(self):
        self.random_state = 20
        self.X = None
        self.y = None

        # RF Feature selected plus sparse cols
        self.cols = ['count', 'diff_srv_rate', 'src_bytes', 'dst_host_srv_count', 'flag', 'dst_bytes', 'serror_rate',
                     'dst_host_diff_srv_rate', 'service', 'dst_host_count', 'dst_host_srv_diff_host_rate', 'logged_in',
                     'protocol_type', 'dst_host_same_src_port_rate', 'hot', 'srv_count', 'wrong_fragment',
                     'num_compromised', 'rerror_rate', 'srv_diff_host_rate', 'urgent', 'num_failed_logins',
                     'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                     'is_guest_login']

        self.qt = QuantileTransformer(output_distribution='normal')
        self.sampler = SMOTE(random_state=0)

    def set_X(self, df):
        self.X = df.loc[:, self.cols]

    def encode_categoricals(self):
        self.X = pd.get_dummies(data=self.X, columns=['protocol_type', 'service', 'flag'])

    def sample(self):
        cols = self.X.columns
        self.X, self.y = self.sampler.fit_resample(self.X, self.y)
        if isinstance(self.X, np.ndarray):
            self.X = pd.DataFrame(self.X, columns=cols)

    def scale(self):
        cols = self.X.columns
        self.X = self.qt.fit_transform(self.X)
        if isinstance(self.X, np.ndarray):
            self.X = pd.DataFrame(self.X, columns=cols)

    def pca_transform(self):
        pca = PCA(n_components=3, random_state=self.random_state)
        self.X = pca.fit_transform(self.X)
        if isinstance(self.X, np.ndarray):
            self.X = pd.DataFrame(self.X, columns=['PCAF1', 'PCAF2', 'PCAF3'])


class Tensor2d_type_1(Tensor2d):
    def __int__(self):
        pass

    def set_y(self, df):
        self.y = df['attack_category']

    def pca(self):
        pass

    def add_target(self):
        self.X['attack_category'] = self.y


class Tensor2d_type_2(Tensor2d):
    def __int__(self):
        pass

    def set_y(self, df):
        self.y = df['attack_category']

    def pca(self):
        Tensor2d.pca_transform(self)

    def add_target(self):
        self.X['attack_category'] = self.y


class Preptensorinputs:
    def __init__(self):
        self.logfile = None
        self.gettrace = getattr(sys, 'gettrace', None)
        self.original_stdout = sys.stdout
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.log_file()

        print(__doc__)

        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.X = None
        self.y = None
        self.full = None

        with timer('\nLoading dataset'):
            self.load_data()

        with timer('\nPreparing Tensor Input Files'):
            for t2d in (Tensor2d_type_1(),
                        Tensor2d_type_2()):
                with timer('\nBuilding 2d tensor - ' + t2d.__class__.__name__):
                    t2d.set_X(self.full)
                    t2d.encode_categoricals()
                    t2d.set_y(self.full)
                    t2d.sample()
                    t2d.scale()
                    t2d.pca()
                    t2d.add_target()
                    self.filehandler.write_csv(self.ds.config['path'], self.ds.config['file'] + '_' +
                                               t2d.__class__.__name__, t2d.X)
                    print('Shape of ' + self.ds.config['file'] + '_' + t2d.__class__.__name__ + ' : ' +
                          str(t2d.X.shape))

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


preptensorinputs = Preptensorinputs()

