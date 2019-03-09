import logging
import pandas as pd
from os import path
from filehandler import Filehandler
from preprocessor import Preprocessor

def process_dataset(filehandler, modeller, dataset_path):
    dataset = filehandler.read_csv(dataset_path)  # Reload dataset to ensure no scaling on scaling successive preds


def main():
    filehandler = Filehandler()
    Preprocessor()


if __name__ == '__main__':
    import logging.config
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    main()
