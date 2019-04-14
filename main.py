from os import path
from contextlib import contextmanager
from dataset import KDDCup1999
from database import Database
from filehandler import Filehandler
from visualize import Visualize
import importlib
import time
import pickle


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


def main():
    ds = KDDCup1999()
    database = Database()
    filehandler = Filehandler()
    visualize = Visualize()
    with timer('Loading dataset'):
        ds.dataset = filehandler.read_csv(ds.config['path'], ds.config['file'])
        ds.set_columns()
    with timer('Transforming dataset'):
        ds.transform()
    with timer('Dataset discovery'):
        ds.discovery()
        ds.drop_duplicates()
        ds.discovery()
    with timer('Linear Separation Tests'):
        #ds.linear()
        #ds.linear_perceptron()
        #ds.linear_svc()
        ds.cluster()
        #ds.boxplot()
        #ds.outliers()
        #ds.disto()
        #ds.heatmap()
        #ds.standardized_data()
        ds.feature_selection_univariate()
    with timer('Setting target'):
        ds.set_target()
    with timer('Sampling'):
        ds.sample('attack_category', 'normal', 52)
    with timer('Encoding dataset'):
        ds.drop_cols()
        ds.onehotencode()
        ds.scale()
    with timer('Persisting transformed dataset and target'):
        filehandler.write_csv(ds.config['path'], ds.config['file'] + '_processed', ds.dataset)
        filehandler.write_csv(ds.config['path'], ds.config['file'] + '_target', ds.target)
    with timer('Modeller'):
        model_module = importlib.import_module('model')
        models = ['RandomForestClf', 'DecisionTreeClf', 'ANNPerceptronClf']
        for m in models:
            cls = getattr(model_module, m)
            model = cls()
            if not model.enabled:
                continue
            print('Processing {}'.format(m))
            model.set_ds(ds.config['path'], ds.config['file'])
            model.fit()
            #model.apply_pca()

            model.score()
            mdb_payload = {
                "name": model.base['stext'],
                "model": pickle.dumps(model.base['model']),
                "scores": model.base['scores'].tolist()
            }

            model.mongo_id = database.insert_one('ml', 'models', mdb_payload)
            model.predict()
            model.base['cm'] = model.get_confusion_matrix()
            mdb_filter = {"_id": model.mongo_id}
            mdb_payload = {"$set":
                         {"cm_tp": int(model.base['cm'][1][1]),
                          "cm_tn": int(model.base['cm'][0][0]),
                          "cm_fp": int(model.base['cm'][0][1]),
                          "cm_fn": int(model.base['cm'][1][0]),
                          "updatedAt": time.time()}}

            database.update_one('ml', 'models', mdb_filter, mdb_payload)

            visualize.confusion_matrix(model.base['cm'], m, ds.config['target'])


if __name__ == '__main__':
    import logging.config
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    with timer('Full model run'):
        main()



# import data

# PREPROCESSING
# feature engineering
# feature selection
# feature transformation
# feature missing values
# outlier handling
# check variable types



# MODEL ASSESSMENT / VALIDATE
# validate partitioning
# choose model metric
# score models


# MODEL/ ALGORITHM
# algorithym from libraries
# select model
# tune hyper parameters
# which algorithyms to run

# collect actual results


