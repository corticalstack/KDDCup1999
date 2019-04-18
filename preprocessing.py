"""
============================================================================
Preprocessing - Initial and extended data discovery with feature engineering
============================================================================
"""
import sys
from contextlib import contextmanager
import time
import numpy as np
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class Preprocessing:

    def __init__(self):
        # Redirect stdout to file for logging
        original_stdout = sys.stdout
        f = open('logs/preprocessing_stdout.txt', 'w')
        sys.stdout = f

        print(__doc__)

        self.filehandler = Filehandler()
        self.visualize = Visualize()
        self.ds = KDDCup1999()

        with timer('\nLoading dataset'):
            self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'])
            self.ds.set_columns()
        with timer('\nTransforming dataset'):
            self.ds.transform()
        with timer('\nInitial dataset discovery'):
            self.ds.shape()
            self.ds.show_duplicates(self.ds.config['level_01'])
            self.ds.drop_duplicates()
            self.show_zeros()
            self.ds.drop_outliers()
            self.ds.shape()
            self.ds.discovery()
        with timer('\nSetting target'):
            self.ds.set_target()
        with timer('\nEvaluating sparse features'):
            self.ds.evaluate_sparse_features(engineer=True)
        with timer('\nVisualising pairplot for selected columns'):
            self.visualize.pairplot(self.ds.dataset, self.ds.config['pairplot_cols'], self.ds.config['pairplot_target'])
        with timer('\nDropping columns'):
            self.ds.drop_cols(self.ds.config['drop_cols_01'])
        with timer('\nEvaluating correlation'):
            self.visualize.correlation_heatmap(self.ds.dataset, title='Correlation Heatmap Before Column Drop')
            self.ds.drop_highly_correlated()
            self.visualize.correlation_heatmap(self.ds.dataset, title='Correlation Heatmap After Column Drop')
        with timer('\nPersisting transformed dataset and target'):
            self.filehandler.write_csv(self.ds.config['path'], self.ds.config['file'] + '_processed', self.ds.dataset)
            self.filehandler.write_csv(self.ds.config['path'], self.ds.config['file'] + '_target', self.ds.target)
            self.ds.shape()

        sys.stdout = original_stdout
        f.close()

    def show_zeros(self):
        df = self.ds.dataset.iloc[:, :-3]
        df[(df == 0)] = np.nan  # Transform 0's to NaN for visualisation of sparseness with missingno
        self.visualize.matrix_missing(df, 'Nullity matrix of features with 0 values')
        self.visualize.bar_missing(df, 'Bar plot of features with 0 values')
        self.visualize.heat_missing(df, 'Heatmap of features with missing values')




preprocessing = Preprocessing()



