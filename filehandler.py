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

    def read_csv2(self, datapath):
        logging.info('Reading file - {}'.format(datapath))
        dataset = pd.read_csv(datapath)
        return dataset

    def read_csv3(self, datapath):
        logging.info('Reading file - {}'.format(datapath))
        dataset = pd.read_csv(datapath, skiprows=1)
        return dataset

    def write_csv(self, folder, file, df):
        full_path = path.join(folder, file)
        df.to_csv(full_path, header=True, index=False)

    def output_scores(self, scores):
        logging.info('Outputing scores file - {}'.format(self.scores_path))
        df = pd.DataFrame(scores)
        df.to_csv(self.scores_path, header=None)

    def output_feature_ranking(self, feature_ranking, target_label, clfname, scoring_variant):
        full_path = self.feature_ranking_path + ' - ' + target_label + ' - ' + clfname + ' - ' + \
                    scoring_variant + '.csv'
        logging.info('Outputing feature ranking file - {}'.format(full_path))
        df = pd.DataFrame(feature_ranking)
        df.index.name = 'Rank'
        df.columns = ['Feature', 'Importance']
        df.to_csv(full_path)
