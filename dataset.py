import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Dataset:

    def __init__(self):
        self.dataset = None
        self.target = None
        self.attack_category = ['normal', 'dos', 'u2r', 'r2l', 'probe']

    def set_columns(self):
        self.dataset.columns = self.config['columns']

    def set_target(self):
        self.target = self.dataset[self.config['target']]

    def shape(self):
        print('--- Shape')
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def row_count_by_target(self, target):
        print('\n--- Row count by {}'.format(target))
        series = self.dataset[target].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.dataset.shape[0]) * 100)))

    def row_target_count_by_group(self, level, by):
        print('\n--- Row count by {}'.format(level))
        df = self.dataset.groupby(level)[by].count()
        df = df.rename(columns={by[0]: 'Count'})
        df['Percent'] = (df['Count'] / self.dataset.shape[0]) * 100
        df_flat = df.reset_index()
        print(df_flat)

    def column_unique_value_count(self):
        print('\n--- Column unique value count')
        for col in self.dataset:
            print('\t{} ({})'.format(col, len(self.dataset[col].unique())))

    def duplicates(self, level):
        print('\n--- Duplicates by {}'.format(level))
        df = self.dataset.groupby(self.dataset.columns.tolist()).size().reset_index(name='duplicates')
        df['duplicates'] = df['duplicates'] - 1
        df_flat = df.groupby(level)[['duplicates']].sum().reset_index()
        print(df_flat)

    def drop_duplicates(self):
        self.dataset.drop_duplicates(keep='first', inplace=True)

    def drop_cols(self):
        self.dataset = self.dataset.drop(columns=self.config['drop_cols'])

    def onehotencode(self):
        self.dataset = pd.get_dummies(self.dataset, columns=self.config['onehotencode_cols'], drop_first=True)

    def scale(self):
        sc = StandardScaler()
        self.dataset = pd.DataFrame(sc.fit_transform(self.dataset), columns=self.dataset.columns)

    def sample(self, level, by, n):
        df_sample = self.dataset[(self.dataset[level] == by)].sample(n)
        print(df_sample.shape)

class KDDCup1999(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.config = {'columns': ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                                   'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                                   'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                                   'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                                   'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                                   'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                                   'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                                   'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'],
                       'path': 'data',
                       'file': 'kddcup.data_10_percent',
                       'target': 'target',
                       'level_01': ['attack_category', 'label'],
                       'drop_cols': ['is_host_login', 'attack_category', 'label', 'target'],
                       'onehotencode_cols': ['protocol_type', 'service', 'flag']}

    def clean(self):
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
        self.dataset['target'] = np.select(conditions, choices, default='na')

    def set_attack_category(self):
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
        self.dataset['attack_category'] = np.select(conditions, self.attack_category, default='na')

    def transform(self):
        self.clean()
        self.set_binary_label()
        self.set_attack_category()

    def discovery(self):
        self.shape()
        self.row_count_by_target(self.config['target'])
        self.row_count_by_target('attack_category')
        self.row_target_count_by_group(self.config['level_01'], [self.config['target']])
        self.column_unique_value_count()
        self.duplicates(self.config['level_01'])
