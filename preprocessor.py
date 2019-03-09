import logging
import pandas as pd
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

    def preprocess(self):
        self.filehandler = Filehandler()
        self.dataset_raw = self.filehandler.read_csv(self.filehandler.data_raw_path)
        logging.info('Original raw dataset loaded - dataset size {}'.format(self.dataset_raw.shape))


