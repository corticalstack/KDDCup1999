import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import ConvexHull


class Dataset:

    def __init__(self):
        self.dataset = None
        self.target = None

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

    def column_zero_count(self):
        print('\n--- Column zero count')
        for col in self.dataset:
            count = (self.dataset[col] == 0).sum()
            print('\t{} {} ({:6.3f}%)'.format(col, count, (count / self.dataset.shape[0]) * 100))

    def duplicates(self, level):
        print('\n--- Duplicates by {}'.format(level))
        df = self.dataset.groupby(self.dataset.columns.tolist()).size().reset_index(name='duplicates')
        df['duplicates'] = df['duplicates'] - 1
        df_flat = df.groupby(level)[['duplicates']].sum().reset_index()
        print(df_flat)

    def drop_duplicates(self):
        self.dataset.drop_duplicates(keep='first', inplace=True)

    def drop_cols(self):
        self.dataset.drop(columns=self.config['drop_cols'])

    def onehotencode(self):
        self.dataset = pd.get_dummies(self.dataset, columns=self.config['onehotencode_cols'], drop_first=True)

    def scale(self):
        sc = StandardScaler()
        self.dataset = pd.DataFrame(sc.fit_transform(self.dataset), columns=self.dataset.columns)

    def sample(self, level, by, n):
        df_sample = self.dataset[(self.dataset[level] == by)].sample(n)
        print(df_sample.shape)

    def linear(self):

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        df_encode = self.dataset.iloc[:, 0:7]
        #df_encode.drop(columns = ['is_host_login', 'num_outbound_cmds', 'attack_category', 'label', 'target'])
        df_encode = df_encode.apply(le.fit_transform)
        sc = StandardScaler()
        mms = MinMaxScaler(feature_range=(0, 1))
        df_encode = pd.DataFrame(mms.fit_transform(df_encode), columns=df_encode.columns)
        #scatter_matrix(df_encode.iloc[:, 0:4], figsize=(15, 11))
        #plt.show()




        print('df encode shape', df_encode.shape)
        print('self dataset shape', self.dataset.shape)
        df_encode = df_encode.set_index(self.dataset.index)
        df_encode['target'] = self.dataset['target']
        cdict = {0: 'red', 1: 'blue'}
        import seaborn as sns
        #sns.set(style="ticks")
        #sns.pairplot(df_encode,  vars=df_encode.columns[:-1], hue="target", palette=cdict, height=5)
        #plt.show()

        #plt.clf()
        #plt.figure(figsize = (10, 6))
        #names = ['normal', 'attack']
        #colors = ['b','r']
        #label = (df_encode.target).astype(np.int)

        #plt.title('Duration vs Protocol Type')
        #plt.xlabel(self.dataset.columns[0])
        #plt.ylabel(self.dataset.columns[1])
        #cdict = {0: 'red', 1: 'blue'}
        #for i in range(len(names)):
        #    bucket = df_encode[df_encode['target'] == i]
        #    bucket = bucket.iloc[:,[0,1]].values
        #    plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i], c = cdict[i])
        #plt.legend()
        #plt.show()

        # Convex hull
        plt.clf()
        plt.figure(figsize=(10, 6))
        names = ['normal', 'attack']
        colors = ['b', 'r']
        label = (df_encode.target).astype(np.int)
        plt.title(self.dataset.columns[0] + ' vs ' + self.dataset.columns[5])
        plt.xlabel(self.dataset.columns[0])
        plt.ylabel(self.dataset.columns[5])
        for i in range(len(names)):
            bucket = df_encode[df_encode['target'] == i]
            bucket = bucket.iloc[:, [0, 5]].values
            hull = ConvexHull(bucket)
            plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i])
            for j in hull.simplices:
                plt.plot(bucket[j, 0], bucket[j, 1], colors[i])
        plt.legend()
        plt.show()

    def linear_perceptron(self):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        df_encode = self.dataset.iloc[:, [0, 5]]
        df_encode = df_encode.apply(le.fit_transform)
        sc = StandardScaler()
        df_encode = pd.DataFrame(sc.fit_transform(df_encode), columns=df_encode.columns)

        print('df encode shape', df_encode.shape)
        print('self dataset shape', self.dataset.shape)
        df_encode = df_encode.set_index(self.dataset.index)
        y = self.dataset['target']

        from sklearn.linear_model import Perceptron
        perceptron = Perceptron(random_state=0)
        perceptron.fit(df_encode, y)
        predicted = perceptron.predict(df_encode)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, predicted)

        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['normal', 'attack']
        plt.title('Perceptron Confusion Matrix - Entire Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN', 'FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
        plt.show()

        from matplotlib.colors import ListedColormap
        plt.clf()
        X_set, y_set = df_encode, y
        y_set = y_set.values
        X_set = X_set.values
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, perceptron.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('navajowhite', 'darkkhaki')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Perceptron Classifier (Decision boundary for Setosa vs the rest)')
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.legend()
        plt.show()

    def linear_svc(self):
        #https: // scikit - learn.org / stable / auto_examples / svm / plot_iris.html
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        df_encode = self.dataset.iloc[:, [0, 5]]
        df_encode = df_encode.apply(le.fit_transform)
        sc = StandardScaler()
        df_encode = pd.DataFrame(sc.fit_transform(df_encode), columns=df_encode.columns)
        y = self.dataset['target']

        from sklearn.svm import SVC
        svm = SVC(C=1.0, kernel='linear', random_state=0)
        svm.fit(df_encode, y)

        predicted = svm.predict(df_encode)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, predicted)

        from matplotlib.colors import ListedColormap
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['normal', 'attack']
        plt.title('SVM Linear Kernel Confusion Matrix - Setosa')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN', 'FP'], ['FN', 'TP']]
        plt.show()

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))

        plt.clf()
        X_set, y_set = df_encode, y
        y_set = y_set.values
        X_set = X_set.values
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('navajowhite', 'darkkhaki')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Perceptron Classifier (Decision boundary for Setosa vs the rest)')
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.legend()
        plt.show()

    def cluster(self):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        df_encode = self.dataset.iloc[:, :-3]
        df_encode = df_encode.apply(le.fit_transform)
        sc = StandardScaler()
        df_encode = pd.DataFrame(sc.fit_transform(df_encode), columns=df_encode.columns)

        df_encode = pca.fit_transform(df_encode)

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=5)
        kmeans.fit(df_encode)
        print(kmeans.cluster_centers_)
        y_km = kmeans.fit_predict(df_encode)
        C = kmeans.cluster_centers_
        plt.scatter(df_encode[y_km == 0, 0], df_encode[y_km == 0, 1], s=100, c='red')
        plt.scatter(df_encode[y_km == 1, 0], df_encode[y_km == 1, 1], s=100, c='black')
        plt.scatter(df_encode[y_km == 2, 0], df_encode[y_km == 2, 1], s=100, c='blue')
        plt.scatter(df_encode[y_km == 3, 0], df_encode[y_km == 3, 1], s=100, c='cyan')
        plt.scatter(df_encode[y_km == 4, 0], df_encode[y_km == 4, 1], s=50, c='green')
        plt.show()

    def boxplot(self):
        import seaborn as sns
        from sklearn import preprocessing
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import minmax_scale
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import RobustScaler
        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.preprocessing import PowerTransformer

        le = preprocessing.LabelEncoder()
        df_encode = self.dataset.iloc[:, :-1]
        df_encode = df_encode.apply(le.fit_transform)
        sc = StandardScaler()
        mms = MinMaxScaler()
        df_encode = pd.DataFrame(mms.fit_transform(df_encode), columns=df_encode.columns)
        df_encode['target'] = self.dataset['target']
        df_encode = df_encode.set_index(self.dataset.index)

        for col in df_encode.columns:
            sns.boxplot(x='target', y=df_encode[col], data=df_encode,
                     palette="vlag")
        #sns.swarmplot(x="target", y=df_encode['duration'], data=df_encode,
        #              size=2, color=".3", linewidth=0)
            plt.show()

    def outliers(self):
        self.dataset.describe()
        for col in self.dataset.columns:
            if self.dataset[col].dtype == np.float64 or self.dataset[col].dtype == np.int64:
                threshold = self.dataset[col].max() * 0.95
                outliers = self.dataset[(self.dataset[col] > 50) & (self.dataset[col] > threshold)]
                if (not outliers.empty) and (len(outliers) < (self.dataset.shape[0] * 0.0001)):
                        print('For column {} deleting {} rows over value {}'.format(col, len(outliers), threshold))
                        self.dataset = pd.concat([self.dataset, outliers]).drop_duplicates(keep=False)


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
                       'drop_cols': ['is_host_login', 'num_outbound_cmds', 'attack_category', 'label', 'target'],
                       'onehotencode_cols': ['protocol_type', 'service', 'flag'],
                       'attack_category': ['normal', 'dos', 'u2r', 'r2l', 'probe']}

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

        print('target type', self.dataset.target.dtype)

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

    def discovery(self):
        self.shape()
        self.row_count_by_target(self.config['target'])
        self.row_count_by_target('attack_category')
        self.row_target_count_by_group(self.config['level_01'], [self.config['target']])
        self.column_unique_value_count()
        self.column_zero_count()
        self.duplicates(self.config['level_01'])
