import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from filehandler import Filehandler


class Preprocessor:

    def __init__(self):
        self.dataset_raw = None
        self.filehandler = None
        self.preprocess()

    @staticmethod
    def impute_numeric_feature_with_zero(dataset):
        dataset.fillna(0, inplace=True)

    @staticmethod
    def impute_categorical_feature_with_blank(dataset):
        dataset.fillna('', inplace=True)

    def data_analysis(self):
        print('_' * 80, "=== Dataset Info ===", sep='\n')
        print('Shape {}'.format(self.dataset_raw.shape))
        print('\n', 'Features', self.dataset_raw.columns.values)
        for col in self.dataset_raw:
            print(col, len(self.dataset_raw[col].unique()))

        print('\n', 'Label Binary Count', self.dataset_raw['label_binary'].value_counts())
        print('\n', 'Label Type Count', self.dataset_raw.groupby('label').label_binary.count(), '\n')

        print(self.dataset_raw[['label', 'duration']].groupby(['label'], as_index=False).mean().sort_values(by='duration',
                                                                                                       ascending=False), '_'*80, sep='\n')
        print(
            self.dataset_raw[['label', 'label_binary','protocol_type']].groupby(['label', 'protocol_type'], as_index=False).count().sort_values(by=['label','protocol_type'],
                                                                                                          ascending=True),
            '_' * 80, sep='\n')

        print(
            self.dataset_raw[['label', 'label_binary','service']].groupby(['label', 'service'], as_index=False).count().sort_values(by=['label','service'],
                                                                                                          ascending=True),
            '_' * 80, sep='\n')

        print(
            self.dataset_raw[['label', 'label_binary', 'flag']].groupby(['label', 'flag'],
                                                                           as_index=False).count().sort_values(
                by=['label', 'flag'],
                ascending=True),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'src_bytes']].groupby(['label'], as_index=False).mean().sort_values(by='src_bytes',
                                                                                                        ascending=False),
          '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'dst_bytes']].groupby(['label'], as_index=False).mean().sort_values(
            by='dst_bytes',
            ascending=False),
              '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'wrong_fragment']].groupby(['label'], as_index=False).mean().sort_values(
            by='wrong_fragment',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'urgent']].groupby(['label'], as_index=False).mean().sort_values(
            by='urgent',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'hot']].groupby(['label'], as_index=False).mean().sort_values(
            by='hot',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_failed_logins']].groupby(['label'], as_index=False).mean().sort_values(
            by='num_failed_logins',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'logged_in']].groupby(['label'], as_index=False).count().sort_values(
            by='logged_in',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_compromised']].groupby(['label'], as_index=False).count().sort_values(
            by='num_compromised',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'root_shell']].groupby(['label'], as_index=False).count().sort_values(
            by='root_shell',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'su_attempted']].groupby(['label'], as_index=False).count().sort_values(
            by='su_attempted',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_root']].groupby(['label'], as_index=False).count().sort_values(
            by='num_root',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_file_creations']].groupby(['label'], as_index=False).count().sort_values(
            by='num_file_creations',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_shells']].groupby(['label'], as_index=False).mean().sort_values(
            by='num_shells',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_access_files']].groupby(['label'], as_index=False).mean().sort_values(
            by='num_access_files',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'num_outbound_cmds']].groupby(['label'], as_index=False).mean().sort_values(
            by='num_outbound_cmds',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'is_host_login']].groupby(['label'], as_index=False).count().sort_values(
            by='is_host_login',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'is_guest_login']].groupby(['label'], as_index=False).count().sort_values(
            by='is_guest_login',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'count']].groupby(['label'], as_index=False).count().sort_values(
            by='count',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'srv_count']].groupby(['label'], as_index=False).count().sort_values(
            by='srv_count',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'serror_rate']].groupby(['label'], as_index=False).mean().sort_values(
            by='serror_rate',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'srv_serror_rate']].groupby(['label'], as_index=False).mean().sort_values(
            by='srv_serror_rate',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'rerror_rate']].groupby(['label'], as_index=False).mean().sort_values(
            by='rerror_rate',
            ascending=False),
            '_' * 80, sep='\n')

        print(self.dataset_raw[['label', 'srv_rerror_rate']].groupby(['label'], as_index=False).mean().sort_values(
            by='srv_rerror_rate',
            ascending=False),
            '_' * 80, sep='\n')





    def clean_label(self):
        self.dataset_raw['label'] = self.dataset_raw['label'].str.rstrip('.')

    def set_binary_label(self):
        conditions = [
            (self.dataset_raw['label'] == 'normal'),
            (self.dataset_raw['label'] == 'back') | (self.dataset_raw['label'] == 'buffer_overflow') |
            (self.dataset_raw['label'] == 'ftp_write') | (self.dataset_raw['label'] == 'guess_passwd') |
            (self.dataset_raw['label'] == 'imap') | (self.dataset_raw['label'] == 'ipsweep') |
            (self.dataset_raw['label'] == 'land') | (self.dataset_raw['label'] == 'loadmodule') |
            (self.dataset_raw['label'] == 'multihop') | (self.dataset_raw['label'] == 'neptune') |
            (self.dataset_raw['label'] == 'nmap') | (self.dataset_raw['label'] == 'perl') |
            (self.dataset_raw['label'] == 'phf') | (self.dataset_raw['label'] == 'pod') |
            (self.dataset_raw['label'] == 'portsweep') | (self.dataset_raw['label'] == 'rootkit') |
            (self.dataset_raw['label'] == 'satan') | (self.dataset_raw['label'] == 'smurf') |
            (self.dataset_raw['label'] == 'spy') | (self.dataset_raw['label'] == 'teardrop') |
            (self.dataset_raw['label'] == 'warezclient') | (self.dataset_raw['label'] == 'warezmaster')
             ]
        choices = ['normal', 'intrusion']
        self.dataset_raw['label_binary'] = np.select(conditions, choices, default='na')

    def preprocess(self):
        self.filehandler = Filehandler()
        self.dataset_raw = self.filehandler.read_csv(self.filehandler.data_raw_path)
        self.clean_label()
        self.set_binary_label()
        self.data_analysis()



