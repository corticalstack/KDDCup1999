import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from mpl_toolkits.mplot3d import Axes3D  # Required for 3d projection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA, KernelPCA


class Visualize:
    def __init__(self):
        self.random_state = 20
        self.class_colours = np.array(['blue', 'red', 'green', 'darkviolet', 'lime', 'darkorange', 'goldenrod',
                                       'cyan', 'silver'])

    @staticmethod
    def confusion_matrix(y, y_pred, title):
        plt.clf()
        df_confusion = pd.crosstab(y, y_pred)
        plt.figure(figsize=(8, 8))
        sns.heatmap(df_confusion, annot=True, annot_kws={"size": 14}, fmt='d', cmap='Greens', cbar=False)
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.savefig(fname='viz/CM - ' + title, dpi=300, format='png')
        plt.show()

    @staticmethod
    def correlation_heatmap(ds, title='Correlation Heatmap', drop=False):
        corr = ds.corr()
        plt.clf()
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
        plt.clf()
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
        title = cola + ' vs ' + colb + ' - label ' + hue
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
        title = 'Convex Hull - ' + cola + ' vs ' + colb + ' - label ' + target
        plt.title(title, fontsize=16)
        plt.xlabel(cola, fontsize=12)
        plt.ylabel(colb, fontsize=12)
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

    @staticmethod
    def kdeplot(title, df, cols):
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 8))

        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        for col in cols:
            sns.kdeplot(df[col], ax=ax)

        plt.savefig(fname='viz/' + 'KDE - ' + title, dpi=300, format='png')
        plt.show()

    @staticmethod
    def matrix_missing(sample_df, title):
        missing_data_df = sample_df.columns[sample_df.isnull().any()].tolist()
        msno.matrix(sample_df[missing_data_df], sparkline=False, fontsize=12, figsize=(30, 22))
        plt.title(title, fontsize=20, y=1.08)
        fig = plt.gcf()
        plt.tight_layout()
        fig.savefig('viz/Nullity - ' + title + '.png')
        plt.show()

    @staticmethod
    def bar_missing(sample_df, title):
        missing_data_df = sample_df.columns[sample_df.isnull().any()].tolist()
        msno.bar(sample_df[missing_data_df], color="black", log=False, figsize=(30, 22))
        plt.title(title, fontsize=24, y=1.05)
        fig = plt.gcf()
        fig.savefig('viz/Nullity - ' + title + '.png')
        plt.show()

    @staticmethod
    def heat_missing(sample_df, title):
        missing_data_df = sample_df.columns[sample_df.isnull().any()].tolist()
        msno.heatmap(sample_df[missing_data_df], figsize=(20, 20))
        plt.title(title, fontsize=24)
        fig = plt.gcf()
        fig.savefig('viz/Nullity - ' + title + '.png')
        plt.show()

    def scatter_clusters(self, df, n_clusters, y_clusters, col_idx, projection=None):
        is_pca = True if 'pca' in col_idx else False
        is_kernelpca = True if 'kpca' in col_idx else False
        is_3d = True if projection == '3d' else False
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=projection)
        df_x = df

        if is_pca:
            n_comp = 3 if is_3d else 2
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            df_x = pca.fit_transform(df_x)

        if is_kernelpca:
            n_comp = 3 if is_3d else 2
            kernelpca = KernelPCA(n_components=n_comp, random_state=self.random_state)
            df_x = kernelpca.fit_transform(df_x)

        if isinstance(df_x, pd.DataFrame):
            df_x = df_x.values

        xlabel = 'PCA Feature 0' if is_pca else 'Kernel PCA Feature 0' if is_kernelpca else df.columns[col_idx[0]]
        ylabel = 'PCA Feature 1' if is_pca else 'Kernel PCA Feature 1' if is_kernelpca else df.columns[col_idx[1]]
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if is_3d:
            title = '3D Cluster PCA' if is_pca else '3D Cluster Kernel PCA' if is_kernelpca else '3D Cluster no PCA'
            zlabel = 'PCA Feature 2' if is_pca else 'Kernel PCA Feature 2' if is_kernelpca else df.columns[col_idx[2]]
            ax.set_zlabel(zlabel, fontsize=12)
        else:
            title = '2D Cluster PCA' if is_pca else '2D Cluster Kernel PCA' if is_kernelpca else '2D Cluster no PCA'

        title = title + ' - ' + str(n_clusters) + ' Clusters'
        title_suffix = ''
        if not is_pca:
            title_suffix = ' - ' + df.columns[col_idx[0]] + ' vs ' + df.columns[col_idx[1]]
            if is_3d:
                title_suffix = title_suffix + ' vs ' + df.columns[col_idx[2]]

        title = title + title_suffix

        if is_pca:
            plt.title(title, fontsize=16)
        else:
            plt.title(title, fontsize=12)

        # 2 clusters minimum
        for c in range(n_clusters):
            if is_3d:
                ax.scatter(df_x[y_clusters == c, col_idx[0]], df_x[y_clusters == c, col_idx[1]],
                           df_x[y_clusters == c, col_idx[2]], alpha=0.2, edgecolors='none', s=30,
                           c=self.class_colours[c])
            else:
                ax.scatter(df_x[y_clusters == c, col_idx[0]], df_x[y_clusters == c, col_idx[1]], alpha=0.2,
                           edgecolors='none', s=30, c=self.class_colours[c])

        plt.savefig(fname='viz/Scatter Cluster- ' + title, dpi=300, format='png')
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
        plt.title(title + ' - ' + cola + ' vs ' + colb, fontsize=16)
        plt.xlabel(cola, fontsize=12)
        plt.ylabel(colb, fontsize=12)
        plt.legend()
        plt.savefig(fname='viz/Boundary - ' + title, dpi=300, format='png')
        plt.show()
