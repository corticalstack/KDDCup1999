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

    def data_analysis(self):
        print('--- Shape')
        print('\tRow count:\t', '{}'.format(self.total_rows))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

        print('\n--- Row count by binary label')
        series = self.dataset['label_binary'].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.total_rows) * 100)))

        print('\n--- Row count by attack group')
        series = self.dataset['attack_group'].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.total_rows) * 100)))

        print('\n--- Row count by attack group/label')
        df = self.dataset.groupby(['attack_group', 'label', ])[['label_binary']].count()
        df = df.rename(columns={'label_binary': 'Count'})
        df['Percent'] = (df['Count'] / self.total_rows) * 100
        df_flat = df.reset_index()
        print(df_flat)

        print('\n--- Column unique value count')
        for col in self.dataset:
            print('\t{} ({})'.format(col, len(self.dataset[col].unique())))

    def data_discovery(self):
        self.total_rows = self.dataset.shape[0]
        print('_' * 40, ' Data Discovery - Full Dataset', '_' * 40, '\n')
        self.data_analysis()

        print('\n--- Duplicates by attack_group/label')
        df = self.dataset.groupby(self.dataset.columns.tolist()).size().reset_index(name='duplicates')
        df['duplicates'] = df['duplicates'] - 1
        df_flat = df.groupby(['attack_group', 'label', ])[['duplicates']].sum().reset_index()
        print(df_flat)

        print('\n', '_' * 40, ' Data Discovery - Dataset Without Duplicates', '_' * 40, '\n')
        self.dataset.drop_duplicates(keep='first', inplace=True)
        self.total_rows = self.dataset.shape[0]
        self.data_analysis()



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
        choices = ['normal', 'intrusion']
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
        self.dataset['attack_group'] = np.select(conditions, choices, default='na')

    def preprocess(self):
        self.filehandler = Filehandler()
        self.dataset = self.filehandler.read_csv(self.filehandler.data_raw_path)
        self.clean_label()
        self.set_binary_label()
        self.set_attack_group()
        self.data_discovery()



