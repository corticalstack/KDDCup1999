import logging
from os import path
import pandas as pd


class Filehandler:
    """Represent constants"""

    def __init__(self):
        self.path_data = 'data'
        self.file_raw_dataset = 'kddcup.data_10_percent'
        self.file_scores = 'scores.csv'
        self.data_raw_path = None
        self.scores_path = None
        self.feature_ranking_path = None
        self.set_paths()

    def set_paths(self):
        self.data_raw_path = path.join(self.path_data, self.file_raw_dataset)
        self.scores_path = path.join(self.path_data, self.file_scores)

    def read_csv(self, datapath, names=None):
        logging.info("Reading file - {}".format(datapath))
        dataset = pd.read_csv(datapath, names=names)
        return dataset

    def output_scores(self, scores):
        logging.info("Outputing scores file - {}".format(self.scores_path))
        df = pd.DataFrame(scores)
        df.to_csv(self.scores_path, header=None)

    def output_feature_ranking(self, feature_ranking, target_label, clfname, scoring_variant):
        full_path = self.feature_ranking_path + ' - ' + target_label + ' - ' + clfname + ' - ' + \
                    scoring_variant + '.csv'
        logging.info("Outputing feature ranking file - {}".format(full_path))
        df = pd.DataFrame(feature_ranking)
        df.index.name = 'Rank'
        df.columns = ['Feature', 'Importance']
        df.to_csv(full_path)
