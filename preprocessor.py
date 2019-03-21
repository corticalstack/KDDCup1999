import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from filehandler import Filehandler


class Preprocessor:

    def __init__(self):
        self.dataset = None
        self.filehandler = None
        self.total_rows = None
        self.preprocess()

    @staticmethod
    def impute_numeric_feature_with_zero(dataset):
        dataset.fillna(0, inplace=True)

    @staticmethod
    def impute_categorical_feature_with_blank(dataset):
        dataset.fillna('', inplace=True)

    def core_analysis(self):
        print('--- Shape')
        print('\tRow count:\t', '{}'.format(self.total_rows))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

        print('\n--- Row count by binary label')
        series = self.dataset['label_binary'].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.total_rows) * 100)))

        print('\n--- Row count by attack category')
        series = self.dataset['attack_category'].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.total_rows) * 100)))

        print('\n--- Row count by attack category/label')
        df = self.dataset.groupby(['attack_category', 'label', ])[['label_binary']].count()
        df = df.rename(columns={'label_binary': 'Count'})
        df['Percent'] = (df['Count'] / self.total_rows) * 100
        df_flat = df.reset_index()
        print(df_flat)

        print('\n--- Column unique value count')
        for col in self.dataset:
            print('\t{} ({})'.format(col, len(self.dataset[col].unique())))

    def evaluate_duplicates(self):
        print('\n--- Duplicates by attack category/label')
        df = self.dataset.groupby(self.dataset.columns.tolist()).size().reset_index(name='duplicates')
        df['duplicates'] = df['duplicates'] - 1
        df_flat = df.groupby(['attack_category', 'label', ])[['duplicates']].sum().reset_index()
        print(df_flat)

    def clean_label(self):
        self.dataset['label'] = self.dataset['label'].str.rstrip('.')

    def set_binary_label(self):
        conditions = [
            (self.dataset['label'] == 'normal'),
            (self.dataset['label'] == 'back') | (self.dataset['label'] == 'buffer_overflow') |
            (self.dataset['label'] == 'ftp_write') | (self.dataset['label'] == 'guess_passwd') |
            (self.dataset['label'] == 'imap') | (self.dataset['label'] == 'ipsweep') |
            (self.dataset['label'] == 'land') | (self.dataset['label'] == 'loadmodule') |
            (self.dataset['label'] == 'multihop') | (self.dataset['label'] == 'neptune') |
            (self.dataset['label'] == 'nmap') | (self.dataset['label'] == 'perl') |
            (self.dataset['label'] == 'phf') | (self.dataset['label'] == 'pod') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'rootkit') |
            (self.dataset['label'] == 'satan') | (self.dataset['label'] == 'smurf') |
            (self.dataset['label'] == 'spy') | (self.dataset['label'] == 'teardrop') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster')
        ]
        choices = [0, 1]
        self.dataset['label_binary'] = np.select(conditions, choices, default='na')

    def set_attack_group(self):
        conditions = [
            (self.dataset['label'] == 'normal'),
            (self.dataset['label'] == 'back') | (self.dataset['label'] == 'land') |
            (self.dataset['label'] == 'neptune') | (self.dataset['label'] == 'pod') |
            (self.dataset['label'] == 'smurf') | (self.dataset['label'] == 'teardrop'),
            (self.dataset['label'] == 'buffer_overflow') | (self.dataset['label'] == 'loadmodule') |
            (self.dataset['label'] == 'perl') | (self.dataset['label'] == 'rootkit'),
            (self.dataset['label'] == 'ftp_write') | (self.dataset['label'] == 'guess_passwd') |
            (self.dataset['label'] == 'imap') | (self.dataset['label'] == 'multihop') |
            (self.dataset['label'] == 'phf') |  (self.dataset['label'] == 'spy') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster'),
            (self.dataset['label'] == 'ipsweep') | (self.dataset['label'] == 'nmap') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'satan')
        ]
        choices = ['normal', 'dos', 'u2r', 'r2l', 'probe']
        self.dataset['attack_category'] = np.select(conditions, choices, default='na')

    def prepare_output_variant_01(self):
        print('prepare output')
        self.dataset.drop('is_host_login', inplace=True, axis=1)

    def corr(self):
        correlation = self.dataset.corr()
        plt.figure(figsize=(40, 40))
        sns.heatmap(correlation, vmax=.8, square=True, annot=True, cmap='cubehelix')

        plt.title('Correlation between different fearures')

        plt.show()

    def preprocess(self):
        self.filehandler = Filehandler()
        self.dataset = self.filehandler.read_csv(self.filehandler.data_raw_path)
        self.clean_label()
        self.set_binary_label()
        self.set_attack_group()

        # Analysis of full dataset
        self.total_rows = self.dataset.shape[0]
        print('_' * 40, ' Data Discovery - Full Dataset', '_' * 40, '\n')
        self.core_analysis()

        # Manage duplicates
        self.evaluate_duplicates()
        self.dataset.drop_duplicates(keep='first', inplace=True)

        # Analysis of dataset without duplicates
        print('\n', '_' * 40, ' Data Discovery - Dataset Without Duplicates', '_' * 40, '\n')
        self.total_rows = self.dataset.shape[0]
        self.core_analysis()

        self.corr()
        self.prepare_output_variant_01()



