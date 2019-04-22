import numpy as np
import pandas as pd


class Dataset:

    def __init__(self):
        self.dataset = None
        self.target = None
        self.column_stats = {}
        self.corr_threshold = 0.80

    def discovery(self):
        self.column_statistics()

    def set_columns(self):
        self.dataset.columns = self.config['columns']

    def set_target(self):
        self.target = self.dataset[self.config['target']]

    def shape(self):
        print('\n--- Shape')
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def column_statistics(self):
        print('\n--- Column Stats')
        for col in self.dataset:
            self.column_stats[col + '_dtype'] = self.dataset[col].dtype
            self.column_stats[col + '_zero_num'] = (self.dataset[col] == 0).sum()
            self.column_stats[col + '_zero_pct'] = (((self.dataset[col] == 0).sum() / self.dataset.shape[0]) * 100)
            self.column_stats[col + '_nunique'] = (self.dataset[col] == 0).nunique()

            print('\n- {} ({})'.format(col, self.column_stats[col + '_dtype']))
            print('\tzero {} ({:.2f}%)'.format(self.column_stats[col + '_zero_num'],
                                               self.column_stats[col + '_zero_pct']))
            print('\tdistinct {}'.format(self.column_stats[col + '_nunique']))

            # Numerical features
            if self.dataset[col].dtype != object:
                self.column_stats[col + '_min'] = (self.dataset[col] == 0).min()
                self.column_stats[col + '_mean'] = (self.dataset[col] == 0).mean()
                self.column_stats[col + '_quantile_25'] = (self.dataset[col] == 0).quantile(.25)
                self.column_stats[col + '_quantile_50'] = (self.dataset[col] == 0).quantile(.50)
                self.column_stats[col + '_quantile_75'] = (self.dataset[col] == 0).quantile(.75)
                self.column_stats[col + '_max'] = (self.dataset[col] == 0).max()
                self.column_stats[col + '_std'] = (self.dataset[col] == 0).std()
                self.column_stats[col + '_skew'] = (self.dataset[col] == 0).skew()
                self.column_stats[col + '_kurt'] = (self.dataset[col] == 0).kurt()
                print('\tmin {}'.format(self.column_stats[col + '_min']))
                print('\tmean {:.3f}'.format(self.column_stats[col + '_mean']))
                print('\t25% {:.3f}'.format(self.column_stats[col + '_quantile_25']))
                print('\t50% {:.3f}'.format(self.column_stats[col + '_quantile_50']))
                print('\t75% {:.3f}'.format(self.column_stats[col + '_quantile_75']))
                print('\tmax {}'.format(self.column_stats[col + '_max']))
                print('\tstd {:.3f}'.format(self.column_stats[col + '_std']))
                print('\tskew {:.3f}'.format(self.column_stats[col + '_skew']))
                print('\tkurt {:.3f}'.format(self.column_stats[col + '_kurt']))

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

    def show_duplicates(self, level):
        print('\n--- Duplicates by {}'.format(level))
        df = self.dataset.groupby(self.dataset.columns.tolist()).size().reset_index(name='duplicates')
        df['duplicates'] = df['duplicates'] - 1
        df_flat = df.groupby(level)[['duplicates']].sum().reset_index()
        print(df_flat)

    def drop_duplicates(self):
        self.dataset.drop_duplicates(keep='first', inplace=True)

    def drop_cols(self, cols):
        print('\n--- Dropping columns')
        print(cols)
        self.dataset.drop(columns=cols, inplace=True)

    # Consider data only less than 95% of max to exlude extreme outliers
    def drop_outliers(self):
        print('\n--- Dropping Outliers')
        for col in self.dataset.columns:
            if self.dataset[col].dtype == np.float64 or self.dataset[col].dtype == np.int64:
                threshold = self.dataset[col].max() * 0.95
                outliers = self.dataset[(self.dataset[col] > 50) & (self.dataset[col] > threshold)]
                if (not outliers.empty) and (len(outliers) < (self.dataset.shape[0] * 0.0001)):
                    print('For column {} deleting {} rows over value {}'.format(col, len(outliers), threshold))
                    self.dataset = pd.concat([self.dataset, outliers]).drop_duplicates(keep=False)

    def drop_highly_correlated(self):
        corr = self.dataset.corr().abs()
        upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        cols_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >=
                                                                           self.corr_threshold)]

        cols_to_drop = list(set(cols_to_drop) - set(self.config['minority_cols_exclude_drop']))
        self.drop_cols(cols_to_drop)


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
                       'drop_cols_01': ['is_host_login', 'num_outbound_cmds', 'label', 'target'],
                       'drop_cols_02': ['attack_category'],
                       'minority_cols_exclude_drop': ['urgent', 'num_failed_logins', 'num_compromised', 'root_shell',
                                                      'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                                                      'num_access_files', 'is_guest_login'],
                       'onehotencode_cols': ['protocol_type', 'service', 'flag'],
                       'attack_category': ['normal', 'dos', 'u2r', 'r2l', 'probe'],
                       'pairplot_cols': ['duration', 'dst_host_diff_srv_rate', 'dst_host_srv_count', 'logged_in',
                                         'serror_rate', 'count'],
                       'pairplot_target': 'target'}

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
        self.dataset['target'] = np.select(conditions, choices, default=0)

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
        self.dataset['attack_category'] = np.select(conditions, self.config['attack_category'], default='na')

    def transform(self):
        self.clean()
        self.set_binary_label()
        self.set_attack_category()

    def evaluate_sparse_features(self, engineer=False):
        print('\n--- Evaluating sparse features')
        for col in self.dataset.columns:
            key = col + '_zero_pct'
            if key in self.column_stats:
                if self.column_stats[key] >= 99:
                    print('\n{} {:.3f}%'.format(col, self.column_stats[key]))
                    self.row_target_count_by_group(['label', 'attack_category', col], ['label'])

        # Handcrafted engineering after column evaluation
        if engineer:
            # Col land - 19 of 20 rows land=1 for attack_category=land, set rare occurrence to 0 (data quality issue?)
            self.dataset.loc[self.dataset['label'] == 'normal', 'land'] = 0

            # Col urgent - rare and appears noisy, not signaling any particular attack type, remove
            # Col su_attempted - rare, occurs only once for intrusion and few times for normal, remove
            self.drop_cols(['urgent', 'su_attempted'])

    def discovery(self):
        Dataset.column_statistics(self)
        self.row_count_by_target(self.config['target'])
        self.row_count_by_target('attack_category')
        self.row_target_count_by_group(self.config['level_01'], [self.config['target']])

