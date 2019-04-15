"""
============================================================================
Preprocessing - Initial and extended data discovery with feature engineering
============================================================================
"""
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
from filehandler import Filehandler
from dataset import KDDCup1999
from visualize import Visualize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.datasets import load_wine
from pandas.plotting import scatter_matrix
from visualize import Visualize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap





@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class LinearSeparability:
    def __init__(self):
        self.filehandler = Filehandler()
        self.visualize = Visualize()
        self.ds = KDDCup1999()
        self.X = None
        self.y = None
        self.random_state = 20
        self.class_colours = np.array(["red", "green", "blue", "black", "cyan"])
        # self.target = None

        # self.scatter_matrix()
        # self.scatter_target()
        # self.convex_hull()
        # self.perceptron()
        # self.svm()
        # self.rbf()



        with timer('\nLoading dataset'):
            self.load_data()
            self.set_data()
        with timer('\nInitial dataset discovery'):
            self.ds.shape()
        with timer('\nScatter target'):
            # self.scatter()
            pass
        with timer('\nConvex hull'):
            # self.convex_hull()
            pass
        with timer('\nPerceptron'):
            pass
            #self.perceptron()
        with timer('\nSVM'):
            pass
            #self.svm()
        with timer('\nRBF'):
            self.rbf()

    def scatter(self):
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='count', hue='attack_category')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='serror_rate', hue='attack_category')
        self.visualize.scatter(self.X, cola='dst_host_srv_count', colb='dst_host_count', hue='attack_category')
        self.visualize.scatter(self.X, cola='rerror_rate', colb='count', hue='attack_category')
        self.visualize.scatter(self.X, cola='srv_diff_host_rate', colb='srv_count', hue='attack_category')

    def convex_hull(self):
        buckets = self.X['attack_category'].unique()
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='dst_host_srv_count', colb='count')
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='dst_host_srv_count', colb='serror_rate')
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='dst_host_srv_count', colb='dst_host_count')
        self.visualize.convex_hull(self.X, buckets, 'attack_category', cola='rerror_rate', colb='count')


    @staticmethod
    def confusion_matrix(y, y_pred):
        cm = confusion_matrix(y, y_pred)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        class_names = ['Negative', 'Positive']
        plt.title('Perceptron Confusion Matrix - Entire Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        s = [['TN', 'FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
        plt.show()


    def load_data(self):
        self.ds.dataset = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_processed')
        self.ds.target = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_target')

    def set_data(self):
        self.X = self.ds.dataset
        self.y = self.ds.target


    def perceptron(self):
        perceptron = Perceptron(max_iter=100, tol=1e-3, random_state=self.random_state)
        _x = self.X.iloc[:, [4, 5]]
        # Boolean cast classes other than 1 to 0
        _y = (self.y == 1).astype(np.int)
        _y = _y.values.ravel()
        perceptron.fit(_x, _y)
        predicted = perceptron.predict(_x)
        #self.confusion_matrix(_y, predicted)
        self.visualize.boundary(_x, _y, perceptron)

    def svm(self):
        plt.clf()
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        _x = self.X.iloc[:, [4, 5]]
        _x = sc.fit_transform(_x)

        svm = LinearSVC(max_iter=500, random_state=self.random_state, tol=1e-5)

        # Boolean cast classes other than 1 to 0
        _y = (self.y == 1).astype(np.int)
        _y = _y.values.ravel()
        svm.fit(_x, _y)
        predicted = svm.predict(_x)
        #self.confusion_matrix(_y, predicted)
        self.visualize.boundary(_x, _y, svm)

    def rbf(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        _x = self.X.iloc[:30000, [4, 7]]
        _x = sc.fit_transform(_x)
        # Boolean cast classes other than 1 to 0
        _y = (self.y == 1).astype(np.int)
        _y = _y[:30000].values.ravel()

        svm = SVC(kernel='rbf', gamma=1.0, C=1.0, random_state=self.random_state)
        svm.fit(_x, _y)
        predicted = svm.predict(_x)
        #self.confusion_matrix(_y, predicted)
        self.visualize.boundary(_x, _y, svm)


def linear(self):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df_encode = self.dataset.iloc[:, 0:7]
    # df_encode.drop(columns = ['is_host_login', 'num_outbound_cmds', 'attack_category', 'label', 'target'])
    df_encode = df_encode.apply(le.fit_transform)
    sc = StandardScaler()
    mms = MinMaxScaler(feature_range=(0, 1))
    df_encode = pd.DataFrame(mms.fit_transform(df_encode), columns=df_encode.columns)

    print('df encode shape', df_encode.shape)
    print('self dataset shape', self.dataset.shape)
    df_encode = df_encode.set_index(self.dataset.index)
    df_encode['target'] = self.dataset['target']
    cdict = {0: 'red', 1: 'blue'}
    import seaborn as sns
    # sns.set(style="ticks")
    # sns.pairplot(df_encode,  vars=df_encode.columns[:-1], hue="target", palette=cdict, height=5)
    # plt.show()

    # plt.clf()
    # plt.figure(figsize = (10, 6))
    # names = ['normal', 'attack']
    # colors = ['b','r']
    # label = (df_encode.target).astype(np.int)

    # plt.title('Duration vs Protocol Type')
    # plt.xlabel(self.dataset.columns[0])
    # plt.ylabel(self.dataset.columns[1])
    # cdict = {0: 'red', 1: 'blue'}
    # for i in range(len(names)):
    #    bucket = df_encode[df_encode['target'] == i]
    #    bucket = bucket.iloc[:,[0,1]].values
    #    plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i], c = cdict[i])
    # plt.legend()
    # plt.show()

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
    # https: // scikit - learn.org / stable / auto_examples / svm / plot_iris.html
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

linearSeparability = LinearSeparability()