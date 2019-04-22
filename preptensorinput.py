"""
===========================================================================
Prepare tensor input files for neural network modelling
===========================================================================
The following prepares input for 2D tensors as input to neural networks
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
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


class Tensor2d:
    def __init__(self):
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

    def set_X(self, df):
        self.X = df.loc[:, self.cols]

    def encode_categoricals(self):
        self.X = pd.get_dummies(data=self.X, columns=['protocol_type', 'service', 'flag'])

    def scale(self):
        return self.qt.fit_transform(self.X)

    def scale(self):
        self.X['target'] = self.y

    def output_file(self, suffix):
        self.filehandler.write_csv(self.ds.config['path'], self.ds.config['file'] + suffix, self.X)


class Tensor2d_type_a(Tensor2d):
    def __int__(self):
        self.sampler = SMOTE(random_state=0)

    def set_y(self, df):
        self.y = df['target']

    def resample(self):
        self.X, self.y = self.sampler.fit_resample(self.X, self.y)

    def output_file(self):
        Tensor2d.output_file('type_a')


class Tensor2d_type_b(Tensor2d):
    pass


class Preparetensorinputs:
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

        with timer('\nLoading dataset'):
            self.load_data()

        with timer('\nPreparing Tensor Input Files'):
            print('Encoding categoricals')

            for t2d in (Tensor2d_type_a(),
                        Tensor2d_type_b()):

                t2d.set_x(self.full)
                t2d.encode_categoricals()
                t2d.set_y(self.full)
                t2d.resample()
                t2d.scale()
                t2d.append_target()
                t2d.output_file()

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


preparetensorinputs = Preparetensorinputs()
