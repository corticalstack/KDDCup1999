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
        fig, ax = plt.subplots(figsize=(100, 100))
        sns.pairplot(ds, vars=cols, hue=hue, palette='hls')
        fig.subplots_adjust(top=1.5, bottom=0.08)
        fig.suptitle(title, size=8, y=1.08)
        plt.savefig(fname='viz/' + title, dpi=300, format='png')
        plt.show()
