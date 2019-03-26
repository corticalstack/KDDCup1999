# Author:   Jon-Paul Boyd
# Date:     16/01/2018
# IMAT5234  Applied Computational Intelligence - Mini Project
# Customer Relationship Management - Predict churn, appetency and upselling on Orange dataset
# Modeller

import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from filehandler import Filehandler


class Modeller:

    def __init__(self):
        self.dataset_train = None
        self.target_train = None
        self.dataset_test = None
        self.target_test = None
        self.test_size = 0.2
        self.random_state = 20
        self.target_label = None
        self.scores = []
        self.score_count = 0
        self.filehandler = None

    def score_baseline_rfc(self):
        self.filehandler = Filehandler()
        X = self.filehandler.read_csv3(self.filehandler.file_dataset)
        y = self.filehandler.read_csv2(self.filehandler.file_target)
        clf = RandomForestClassifier(random_state=self.random_state)
        clf.fit(X, y)

        scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
        predictions = cross_val_predict(clf, X, y, cv=10)
        cm = confusion_matrix(y, predictions)
        print(
            "Confusion matrix for {} - TN {}  FN {}  TP {}  FP {}".format('RFC', cm[0][0], cm[1][0],
                                                                             cm[1][1], cm[0][1]))
        print('finished')



# Do I want to create a class called model that can
# set the model name in __init__
# fit the model
# score the model
# cross val the model
# ie the general behaviour
# I can then specialize ift for any specific model ie create subclases that customize the bits of behavours per model trype
# Have a look at design patterns (frameworks) for Python




