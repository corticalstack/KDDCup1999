import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualize:
    def __init__(self):
        pass

    def correlation_heatmap(self, ds, title='Correlation Heatmap', drop=False):
        corr = ds.corr()
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_title(title, size=16)
        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        dropSelf = np.zeros_like(corr)  # Drop self-correlations
        dropSelf[np.triu_indices_from(dropSelf)] = True
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig(fname='viz/' + title, dpi=300, format='png')
        plt.show()

    def pairplot(self, ds, cols, hue, title='Pairplot'):
        fig, ax = plt.subplots(figsize=(80, 80))
        sns.pairplot(ds, vars=cols, hue=hue, palette='hls')
        fig.subplots_adjust(top=1.5, bottom=0.08)
        fig.suptitle(title, size=8, y=1.08)
        plt.savefig(fname='viz/' + title, dpi=300, format='png')
        plt.show()

    def scatter(self, df, cola, colb, hue):
        # plt.clf()
        # plt.figure(figsize=(10, 6))
        # names = self.X['attack_category'].unique()
        # plt.title(cola + ' vs ' + colb)
        # plt.xlabel(cola)
        # plt.ylabel(colb)
        # for i in range(len(names)):
        #     bucket = self.X[self.X[target] == names[i]]
        #     bucket = bucket.iloc[:, [self.X.columns.get_loc(cola), self.X.columns.get_loc(colb)]].values
        #     plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i], c=self.class_colours[i])
        # plt.legend()
        # plt.show()

        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.title(cola + ' vs ' + colb)
        plt.xlabel(cola)
        plt.ylabel(colb)
        sns.scatterplot(x=cola, y=colb, hue=hue, palette='hls', size=30, alpha=0.4, data=df)
        plt.show()
