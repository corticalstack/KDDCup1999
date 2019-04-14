"""
====================================================
Clustering using KDD Cup 1999 IDS dataset
====================================================
Clustering in 2D and 3D with and without PCA applied
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Clustering:
    def __init__(self):
        self.sc = StandardScaler()
        self.n_clusters = 5
        self.colours = ['black', 'blue', 'red', 'cyan', 'green']
        self.ac_count = {}
        self.weight = 1.0  # Full run 1, testing 0.1, 0.3 etc
        self.random_state = 20
        self.dataset = None
        self.alpha = 0.3
        self.s = 30
        self.x = None
        self.y = None
        self.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
        self.attack_category_int = [0, 1, 2, 3, 4]
        self.attack_category = ['normal', 'dos', 'u2r', 'r2l', 'probe']

        self.feature_idx = {0: 0, 1: 0, 2: 0}
        self.pca_idx = {0: 0, 1: 1, 2: 2, 'pca': True}

        self.load_data()
        self.set_columns()

        # Drop large number of duplicates in dataset
        self.drop_duplicates()

        # Clean up the label column data
        self.clean()

        # Set binary target label
        self.set_binary_label()

        # Set attack_category to more clearly see the majority/minority classes - there are 5 "classes"
        self.set_attack_category()

        # Set original count by attack category
        self.set_attack_category_count()

        # Deal with outliers
        self.outliers()

        # Re-weight dataset (reduce samples per attack category, specifically for testing purposes)
        self.dataset = self.weight_attack_category(self.weight)

        self.set_x_y(self.dataset)
        self.encode()
        self.set_indexes()

        # 2D clustering without PCA
        self.cluster(idx=self.feature_idx)

        # 2D clustering with PCA
        self.cluster(idx=self.pca_idx)

        # 3D clustering without PCA
        self.cluster(idx=self.feature_idx, projection='3d')

        # 3D clustering with PCA
        self.cluster(idx=self.pca_idx, projection='3d')

    def cluster(self, idx, projection=None):
        is_pca = True if 'pca' in idx else False
        is_3d = True if projection == '3d' else False
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=projection)

        df_x = self.x

        if is_pca:
            n_comp = 3 if is_3d else 2
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            df_x = pca.fit_transform(df_x)

        if isinstance(df_x, pd.DataFrame):
            df_x = df_x.values

        if is_3d:
            title = '3D clustering with PCA' if is_pca else '3D clustering without PCA'
            ax.set_xlabel('PCA Feature 0') if is_pca else ax.set_xlabel(self.x.columns[idx[0]])
            ax.set_ylabel('PCA Feature 1') if is_pca else ax.set_ylabel(self.x.columns[idx[1]])
            ax.set_zlabel('PCA Feature 2') if is_pca else ax.set_zlabel(self.x.columns[idx[2]])
        else:
            title = '2D clustering with PCA' if is_pca else '2D clustering without PCA'
            ax.set_xlabel('PCA Feature 0') if is_pca else ax.set_xlabel(self.x.columns[idx[0]])
            ax.set_ylabel('PCA Feature 1') if is_pca else ax.set_ylabel(self.x.columns[idx[1]])

        plt.title(title)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(df_x)
        y_km = kmeans.fit_predict(df_x)

        for c in range(self.n_clusters):
            if is_3d:
                ax.scatter(df_x[y_km == c, idx[0]], df_x[y_km == c, idx[1]], df_x[y_km == c, idx[2]], alpha=self.alpha,
                           edgecolors='none', s=self.s, c=self.colours[c])
            else:
                ax.scatter(df_x[y_km == c, idx[0]], df_x[y_km == c, idx[1]], alpha=self.alpha, edgecolors='none',
                           s=self.s, c=self.colours[c])
        plt.show()

    def load_data(self):
        self.dataset = pd.read_csv('kddcup.data_10_percent')
        print('--- Original Shape')
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def set_columns(self):
        self.dataset.columns = self.columns

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
            (self.dataset['label'] == 'phf') | (self.dataset['label'] == 'spy') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster'),
            (self.dataset['label'] == 'ipsweep') | (self.dataset['label'] == 'nmap') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'satan')
        ]
        self.dataset['attack_category'] = np.select(conditions, self.attack_category, default='na')
        self.dataset['attack_category_int'] = np.select(conditions, self.attack_category_int, default=0)

    def set_attack_category_count(self):
        ac = self.dataset['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value


    def set_x_y(self, ds):
        self.x = ds.iloc[:, :-4]
        self.y = ds.iloc[:, -1].values

    def encode(self):
        le = preprocessing.LabelEncoder()
        self.x = self.dataset.iloc[:, :-4]
        self.x = self.x.apply(le.fit_transform)
        self.x = pd.DataFrame(self.sc.fit_transform(self.x), columns=self.x.columns)

    def set_indexes(self):
        self.feature_idx[0] = self.dataset.columns.get_loc('duration')
        self.feature_idx[1] = self.dataset.columns.get_loc('src_bytes')
        self.feature_idx[2] = self.dataset.columns.get_loc('dst_bytes')


clustering = Clustering()