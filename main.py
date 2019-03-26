from os import path
import time
from contextlib import contextmanager
from dataset import KDDCup1999
from filehandler import Filehandler
from modeller import Modeller


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


def main():
    dataset = KDDCup1999()
    filehandler = Filehandler()
    modeller = Modeller()
    with timer('Loading dataset'):
        dataset.dataset = filehandler.read_csv(dataset.config['path'], dataset.config['file'])
        dataset.set_columns()
    with timer('Transforming dataset'):
        dataset.transform()
    with timer('Dataset discovery'):
        dataset.discovery()
        dataset.drop_duplicates()
        dataset.discovery()
    with timer('Setting target'):
        dataset.set_target()
    with timer('Encoding dataset'):
        dataset.drop_cols()
        dataset.onehotencode()
        dataset.scale()
    with timer('Persisting transformed dataset and target'):
        filehandler.write_csv(dataset.config['path'], dataset.config['file'] + '_processed', dataset.dataset)
        filehandler.write_csv(dataset.config['path'], dataset.config['file'] + '_target', dataset.target)
    with timer('Modeller'):
        #think about builing new model or getting from db as an example
        modeller.score_baseline_rfc()


if __name__ == '__main__':
    import logging.config
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    with timer('Full model run'):
        main()
