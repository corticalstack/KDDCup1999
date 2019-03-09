import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from filehandler import Filehandler


class Preprocessor:

    def __init__(self):
        self.dataset_raw = None
        self.filehandler = None
        self.preprocess()

    @staticmethod
    def impute_numeric_feature_with_zero(dataset):
        dataset.fillna(0, inplace=True)

    @staticmethod
    def impute_categorical_feature_with_blank(dataset):
        dataset.fillna('', inplace=True)

    def data_analysis(self):
        print('_' * 80, "=== Dataset Info ===", sep='\n')
        print('Shape {}'.format(self.dataset_raw.shape))
        print(self.dataset_raw.columns.values, '_' * 80, sep='\n')
        print(self.dataset_raw.info(), '_' * 80, sep='\n')
        print(self.dataset_raw.head(), '_' * 80, sep='\n')
        print(self.dataset_raw.tail(), '_' * 80, sep='\n')
        print(self.dataset_raw.describe(), '_' * 80, sep='\n')
        print(self.dataset_raw.describe(include=['O']), '_' * 80, sep='\n')

    def clean_label(self):
        self.dataset_raw['label'] = self.dataset_raw['label'].str.rstrip('.')

    def set_binary_label(self):
        conditions = [
            (self.dataset_raw['label'] == 'normal'),
            (self.dataset_raw['label'] == 'back') | (self.dataset_raw['label'] == 'buffer_overflow') |
            (self.dataset_raw['label'] == 'ftp_write') | (self.dataset_raw['label'] == 'guess_passwd') |
            (self.dataset_raw['label'] == 'imap') | (self.dataset_raw['label'] == 'ipsweep') |
            (self.dataset_raw['label'] == 'land') | (self.dataset_raw['label'] == 'loadmodule') |
            (self.dataset_raw['label'] == 'multihop') | (self.dataset_raw['label'] == 'neptune') |
            (self.dataset_raw['label'] == 'nmap') | (self.dataset_raw['label'] == 'perl') |
            (self.dataset_raw['label'] == 'phf') | (self.dataset_raw['label'] == 'pod') |
            (self.dataset_raw['label'] == 'portsweep') | (self.dataset_raw['label'] == 'rootkit') |
            (self.dataset_raw['label'] == 'satan') | (self.dataset_raw['label'] == 'smurf') |
            (self.dataset_raw['label'] == 'spy') | (self.dataset_raw['label'] == 'teardrop') |
            (self.dataset_raw['label'] == 'warezclient') | (self.dataset_raw['label'] == 'warezmaster')
             ]
        choices = ['normal', 'intrusion']
        self.dataset_raw['label_binary'] = np.select(conditions, choices, default='na')

    def preprocess(self):
        self.filehandler = Filehandler()
        self.dataset_raw = self.filehandler.read_csv(self.filehandler.data_raw_path)
        self.clean_label()
        self.set_binary_label()
        self.data_analysis()



