import logging
import pandas as pd
from os import path
import time
from contextlib import contextmanager
from filehandler import Filehandler
from preprocessor import Preprocessor
from modeller import Modeller


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


def process_dataset(filehandler, modeller, dataset_path):
    dataset = filehandler.read_csv(dataset_path)  # Reload dataset to ensure no scaling on scaling successive preds


def main():
    dataset = None
    filehandler = Filehandler()
    modeller = Modeller()
    with timer('Preprocessor'):
        Preprocessor()
    with timer('Modeller'):
        modeller.score_baseline_rfc()



if __name__ == '__main__':
    import logging.config
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    with timer('Full model run'):
        main()
