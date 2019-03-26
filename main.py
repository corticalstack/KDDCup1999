from os import path
import time
from contextlib import contextmanager
from dataset import KDDCup1999
from filehandler import Filehandler
from visualize import Visualize
import importlib


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


def main():
    dataset = KDDCup1999()
    filehandler = Filehandler()
    visualize = Visualize()
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
        model_module = importlib.import_module('model')
        models = ['RandomForestClf', 'DecisionTreeClf', 'ANNPerceptronClf']
        for m in models:
            cls = getattr(model_module, m)
            model = cls()
            if not model.enabled:
                continue
            print('Processing {}'.format(m))
            model.set_dataset(dataset.config['path'], dataset.config['file'])
            model.fit()
            model.score()
            model.predict()
            visualize.confusion_matrix(model.get_confusion_matrix(), m, dataset.config['target'])


if __name__ == '__main__':
    import logging.config
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    with timer('Full model run'):
        main()
