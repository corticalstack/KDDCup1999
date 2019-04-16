import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix
import itertools

class Visualize:
    def __init__(self):
        self.class_colours = np.array(["red", "green", "blue", "black", "cyan"])

    @staticmethod
    def confusion_matrix(y, y_pred, title, class_names):
        cm = confusion_matrix(y, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('tab20c'))
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(fname='viz/CM - ' + title, dpi=300, format='png')
        plt.show()

    @staticmethod
    def correlation_heatmap(ds, title='Correlation Heatmap', drop=False):
        corr = ds.corr()
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_title(title, size=16)
        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        dropSelf = np.zeros_like(corr)  # Drop self-correlations
        dropSelf[np.triu_indices_from(dropSelf)] = True
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig(fname='viz/CorrHeatmap - ' + title, dpi=300, format='png')
        plt.show()

    @staticmethod
    def pairplot(ds, cols, hue, title='Pairplot'):
        fig, ax = plt.subplots(figsize=(80, 80))
        sns.pairplot(ds, vars=cols, hue=hue, palette='hls')
        fig.subplots_adjust(top=1.5, bottom=0.08)
        fig.suptitle(title, size=8, y=1.08)
        plt.savefig(fname='viz/Pairplot - ' + title, dpi=300, format='png')
        plt.show()

    @staticmethod
    def scatter(df, cola, colb, hue):
        plt.clf()
        df[hue] = df[hue].astype('category')
        plt.figure(figsize=(10, 6))
        title = cola + ' vs ' + colb
        plt.title(title, fontsize=16)
        plt.xlabel(cola, fontsize=12)
        plt.ylabel(colb, fontsize=12)
        sns.scatterplot(x=cola, y=colb, hue=hue, palette='Set1', legend=False, size=30, alpha=0.4, data=df)
        plt.savefig(fname='viz/Scatter - ' + title, dpi=300, format='png')
        plt.show()

    @staticmethod
    def convex_hull(df, buckets, target, cola, colb):
        cmap = plt.get_cmap('Set1')
        plt.clf()
        plt.figure(figsize=(10, 6))
        title = 'Convex Hull - ' + cola + ' vs ' + colb
        plt.title(title)
        plt.xlabel(cola)
        plt.ylabel(colb)
        for i in range(len(buckets)):
            bucket = df[df[target] == buckets[i]]
            bucket = bucket.iloc[:, [df.columns.get_loc(cola), df.columns.get_loc(colb)]].values
            hull = ConvexHull(bucket)
            hull_color = tuple(np.atleast_2d(cmap(i/10.))[0])
            plt.scatter(bucket[:, 0], bucket[:, 1], label=buckets[i], c=np.atleast_2d(cmap(i/10.)), alpha=0.4)
            for j in hull.simplices:
                plt.plot(bucket[j, 0], bucket[j, 1], color=hull_color)
        plt.legend()
        plt.savefig(fname='viz/' + title, dpi=300, format='png')
        plt.show()

    def boundary(self, x, y, clf, title, cola, colb):
        plt.clf()
        if isinstance(x, pd.DataFrame):
            x_set, y_set = x.values, y
        else:
            x_set, y_set = x, y

        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
        plt.contourf(x1, x2, clf.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                     alpha=0.4, cmap=plt.get_cmap('Set1'))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                        c=self.class_colours[i], label=j)
        plt.title(title + ' - ' + cola + ' vs ' + colb)
        plt.xlabel(cola)
        plt.ylabel(colb)
        plt.legend()
        plt.savefig(fname='viz/Boundary - ' + title, dpi=300, format='png')
        plt.show()
