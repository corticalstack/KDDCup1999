"""
==============================================================================
File handler
==============================================================================
File handler
"""
import logging
from os import path
import pandas as pd


class Filehandler:
    """Represent constants"""

    def __init__(self):
        self.file_target = 'target'
        self.file_scores = 'scores.csv'
        self.data_raw_path = None
        self.scores_path = None
        self.feature_ranking_path = None

    def read_csv(self, folder, file):
        full_path = path.join(folder, file)
        logging.info('Reading file - {}'.format(full_path))
        return pd.read_csv(full_path)

    def write_csv(self, folder, file, df):
        full_path = path.join(folder, file)
        df.to_csv(full_path, header=True, index=False)

