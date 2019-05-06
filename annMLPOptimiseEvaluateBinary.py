"""
===========================================================================
Logistic Regression and evaluation
===========================================================================
Logistic Regression
"""
import sys
from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from filehandler import Filehandler
from dataset import KDDCup1999
import matplotlib.pyplot as plt
import seaborn as sns


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class AnnMLPOptimiseEvaluate:
    def __init__(self):
        # self.logfile = None
        # self.gettrace = getattr(sys, 'gettrace', None)
        # self.original_stdout = sys.stdout
        # self.timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.log_file()

        print(__doc__)

        self.n_classes = 5
        self.random_state = 20
        self.filehandler = Filehandler()
        self.ds = KDDCup1999()
        self.folder = 'viz'
        self.fprefix_binary = 'Hyper - annMLPBinary - '
        self.fprefix_multi = 'Hyper - annMLPMulti - '

        # Datasets
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.hyp = None
        self.lr = None
        self.label_map_int_2_string = {0: 'good', 1: 'bad', '0': 'good', '1': 'bad'}
        self.label_map_string_2_int = {'normal': 0, 'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4}
        self.max_iters = 100

        with timer('\nPreparing dataset'):
            self.load_data()
            self.set_y()
            self.remove_target_from_X()
            self.train_test_split()

        with timer('\nPreparing base logistic regression'):
            self.lr = LogisticRegression(max_iter=self.max_iters)
            self.lr.fit(self.X_train, self.y_train)

        with timer('\nPreparing confusion matrix and base DR'):
            self.y_pred = self.lr.predict(self.X_test)
            cm = confusion_matrix(self.y_test, self.y_pred)
            self.tp = self.get_tp_from_cm(cm)
            self.tn = self.get_tn_from_cm(cm)
            self.fp = self.get_fp_from_cm(cm)
            self.fn = self.get_fn_from_cm(cm)
            self.dr = self.tp / (self.tp + self.fp)
            print('log reg dr', self.dr)

        with timer('\nVisualising optimisation search'):
            self.load_hyp()
            self.hyp['lr'] = round(self.hyp['lr'] / 1000, 3)

            # Hyperparameter correlation with val DR
            self.hyp_val_dr = self.hyp
            self.hyp_val_dr.drop(['round_epochs', 'epochs', 'loss', 'dr', 'far', 'acc', 'val_loss', 'val_acc', 'val_far'], axis=1, inplace=True)
            self.dr_corr = self.hyp_val_dr.corr()
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 10))
            title = 'Validation DR Hyperparameter Correlation'
            ax.set_title(title, size=16)
            colormap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(self.dr_corr, cmap=colormap, annot=True, fmt=".2f", cbar=False, vmin=-0.4, vmax=0.4)
            plt.xticks(range(len(self.dr_corr.columns)), self.dr_corr.columns)
            plt.yticks(range(len(self.dr_corr.columns)), self.dr_corr.columns)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            self.hyp['val_dr_change'] = round(self.hyp.val_dr - self.dr, 3)
            pd.set_option('display.max_columns', 100)
            print(self.hyp.sort_values(by='val_dr', ascending=False).head())

            self.color = 'cornflowerblue'

            metric = 'lr'
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.boxplot(x=metric, y='val_dr_change', data=self.hyp.reset_index(), color=self.color)
            title = 'Validation DR Change Over Baseline As Fn Of Learning Rate'
            plt.title(title, fontsize=16)
            plt.xlabel('Learning Rate', fontsize=12)
            plt.ylabel('Validation DR Change', fontsize=12)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            metric = 'first_neuron'
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.boxplot(x=metric, y='val_dr_change', data=self.hyp.reset_index(), color=self.color)
            title = 'Validation DR Change Over Baseline As Fn Of # Neurons First Layer'
            plt.title(title, fontsize=16)
            plt.xlabel('First Neuron', fontsize=12)
            plt.ylabel('Validation DR Change', fontsize=12)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            metric = 'hidden_layers'
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.boxplot(x=metric, y='val_dr_change', data=self.hyp.reset_index(), color=self.color)
            title = 'Validation DR Change Over Baseline As Fn Of # Hidden Layers'
            plt.title(title, fontsize=16)
            plt.xlabel('Hidden Layers', fontsize=12)
            plt.ylabel('Validation DR Change', fontsize=12)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            metric = 'hidden_neuron'
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.boxplot(x=metric, y='val_dr_change', data=self.hyp.reset_index(), color=self.color)
            title = 'Validation DR Change Over Baseline As Fn Of # Hidden Layer Neurons'
            plt.title(title, fontsize=16)
            plt.xlabel('Hidden Neurons', fontsize=12)
            plt.ylabel('Validation DR Change', fontsize=12)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            metric = 'batch_size'
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.boxplot(x=metric, y='val_dr_change', data=self.hyp.reset_index(), color=self.color)
            title = 'Validation DR Change Over Baseline As Fn Of Batch Size'
            plt.title(title, fontsize=16)
            plt.xlabel('Batch Size', fontsize=12)
            plt.ylabel('Validation DR Change', fontsize=12)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            metric = 'dropout'
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 8))
            ax = sns.boxplot(x=metric, y='val_dr_change', data=self.hyp.reset_index(), color=self.color)
            title = 'Validation DR Change Over Baseline As Fn Of Dropout'
            plt.title(title, fontsize=16)
            plt.xlabel('Dropout', fontsize=12)
            plt.ylabel('Validation DR Change', fontsize=12)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            plt.clf()
            fig, ax = plt.subplots(figsize=(9, 7))
            df_grid = self.hyp.reset_index().groupby(['first_neuron', 'hidden_neuron']).val_dr_change.mean().unstack()
            ax = sns.heatmap(data=df_grid, cmap=(sns.diverging_palette(10, 220, sep=80, n=7)), annot=True, cbar=False)
            title = 'Validation DR Change Over Baseline As Fn Of First Neuron & Hidden Neuron'
            plt.title(title, fontsize=12)
            plt.xlabel('Hidden Neuron', fontsize=10)
            plt.ylabel('First Neuron', fontsize=10)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            plt.clf()
            fig, ax = plt.subplots(figsize=(9, 7))
            df_grid = self.hyp.reset_index().groupby(['hidden_layers', 'hidden_neuron']).val_dr_change.mean().unstack()
            ax = sns.heatmap(data=df_grid, cmap=(sns.diverging_palette(10, 220, sep=80, n=7)), annot=True, cbar=False)
            title = 'Validation DR Change Over Baseline As Fn Of Hidden Layers & Hidden Neuron'
            plt.title(title, fontsize=16)
            plt.xlabel('Hidden Neuron', fontsize=10)
            plt.ylabel('Hidden Layers', fontsize=10)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            plt.clf()
            fig, ax = plt.subplots(figsize=(9, 7))
            df_grid = self.hyp.reset_index().groupby(['batch_size', 'dropout']).val_dr_change.mean().unstack()
            ax = sns.heatmap(data=df_grid, cmap=(sns.diverging_palette(10, 220, sep=80, n=7)), annot=True, cbar=False)
            title = 'Validation DR Change Over Baseline As Fn Of Batch Size & Dropout'
            plt.xlabel('Dropout', fontsize=10)
            plt.ylabel('Batch Size', fontsize=10)
            plt.title(title, fontsize=16)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

            plt.clf()
            fig, ax = plt.subplots(figsize=(9, 7))
            df_grid = self.hyp.reset_index().groupby(['lr', 'dropout']).val_dr_change.mean().unstack()
            ax = sns.heatmap(data=df_grid, cmap=(sns.diverging_palette(10, 220, sep=80, n=7)), annot=True, cbar=False)
            title = 'Validation DR Change Over Baseline As Fn Of Learning Rate & Dropout'
            plt.xlabel('Dropout', fontsize=10)
            plt.ylabel('Learning Rate', fontsize=10)
            plt.title(title, fontsize=16)
            plt.savefig(fname=self.fname(title), dpi=300, format='png')
            plt.show()

        #self.log_file()
        print('Finished')

    def get_base_dr(self):
        y_pred = pd.Series(0.5, index=self.y_train.index)
        cm = confusion_matrix(self.y_train, y_pred)
        tp = self.get_tp_from_cm(cm)
        fn = self.get_fn_from_cm(cm)
        dr = tp / (tp + fn)
        print('dr ', dr)
        return dr

    # True positives are the diagonal elements
    def get_tp_from_cm(self, cm):
        tp = np.diag(cm)
        print('tp', np.sum(np.diag(cm)))
        return np.sum(tp)

    def get_tn_from_cm(self, cm):
        tn = []
        for i in range(self.n_classes):
            temp = np.delete(cm, i, 0)  # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            tn.append(sum(sum(temp)))
        print('tn ', np.sum(tn))
        return np.sum(tn)

    # Sum of columns minus diagonal
    def get_fp_from_cm(self, cm):
        fp = []
        for i in range(self.n_classes):
            fp.append(sum(cm[:, i]) - cm[i, i])
        print('fp ', np.sum(fp))
        return np.sum(fp)

    # Sum of rows minus diagonal
    def get_fn_from_cm(self, cm):
        fn = []
        for i in range(self.n_classes):
            fn.append(sum(cm[i, :]) - cm[i, i])
        print('fn', np.sum(fn))
        return np.sum(fn)

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
        self.X = self.filehandler.read_csv(self.ds.config['path'], self.ds.config['file'] + '_Tensor2d_type_1')
        print('\tRow count:\t', '{}'.format(self.X.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.X.shape[1]))

    def load_hyp(self):
        self.hyp = pd.read_csv('tuning/Hyperparameter tuning - AnnMLPMultiOptimize_1.csv')

    def set_y(self):
        self.y = self.X['attack_category']
        self.y = self.y.map(self.label_map_string_2_int)

    def remove_target_from_X(self):
        self.X.drop('attack_category', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def map_target_to_label(self, t):
        return np.vectorize(self.label_map_int_2_string.get)(t)

    def fname(self, title):
        return '{}/{}.png'.format(self.folder, self.fprefix_multi + title)


annmlpoptimiseevaluate = AnnMLPOptimiseEvaluate()



