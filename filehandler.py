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

    def read_csv(self, datapath):
        logging.info("Reading file - {}".format(datapath))
        dataset = pd.read_csv(datapath, names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                                        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                                        "num_compromised", "root_shell", "su_attempted", "num_root",
                                        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                                        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                                        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                                        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                                        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                                        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                                        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                                        "dst_host_srv_rerror_rate", "label"])
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
