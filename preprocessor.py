import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


class Preprocessor:

    def __init__(self):
        self.target = None
        self.total_rows = None
        self.preprocess()






    def prepare_output_variant_01(self):
        print('prepare output')
        self.target = self.dataset['target']
        cols = ['is_host_login', 'attack_category', 'label', 'target']
        self.dataset = self.dataset.drop(columns=cols)
        #self.dataset.drop('is_host_login', inplace=True, axis=1)

        # One hot encode
        self.dataset = pd.get_dummies(self.dataset, columns=['protocol_type', 'service', 'flag'], drop_first=True)

        # Scale
        sc_X = StandardScaler()
        self.dataset = pd.DataFrame(sc_X.fit_transform(self.dataset), columns=self.dataset.columns)
        print('finished')



    def corr(self):
        correlation = self.dataset.corr()
        plt.figure(figsize=(40, 40))
        sns.heatmap(correlation, vmax=.8, square=True, annot=True, cmap='cubehelix')

        plt.title('Correlation between different fearures')

        plt.show()

    def preprocess(self):


        # Analysis of full dataset
        self.total_rows = self.dataset.shape[0]
        print('_' * 40, ' Data Discovery - Full Dataset', '_' * 40, '\n')
        self.core_analysis()

        # Manage duplicates
        self.evaluate_duplicates()
        self.dataset.drop_duplicates(keep='first', inplace=True)

        # Analysis of dataset without duplicates
        print('\n', '_' * 40, ' Data Discovery - Dataset Without Duplicates', '_' * 40, '\n')
        self.total_rows = self.dataset.shape[0]
        self.core_analysis()

        self.corr()
        self.prepare_output_variant_01()
        self.dataset.to_csv(self.filehandler.file_dataset, index=False)
        self.target.to_csv(self.filehandler.file_target, index=False)



