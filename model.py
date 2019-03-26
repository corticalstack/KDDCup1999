from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from filehandler import Filehandler


class Model:

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

    def set_dataset(self):

    def set_target(self):

    def set_model(self):
        self.set_model()

    def get_confusion_matrix(self):
        return confusion_matrix(y, self.predictions)

    def print_confusion_matrix(self):
        print(
            "Confusion matrix for {} - TN {}  FN {}  TP {}  FP {}".format('RFC', cm[0][0], cm[1][0],
                                                                             cm[1][1], cm[0][1]))



class RandomForestClf(Model):

    def set_model(self):
        self.model = RandomForestClassifier(random_state=self.random_state)

    def fit_model(self):
        self.model.fit(X, y)

    def score_model(self):
        scores = cross_val_score(self.model, X, y, cv=10, scoring='roc_auc')

    def get_predictions(self):
        self.predictions = cross_val_predict(self.model, X, y, cv=10)





# Do I want to create a class called model that can
# set the model name in __init__
# fit the model
# score the model
# cross val the model
# ie the general behaviour
# I can then specialize ift for any specific model ie create subclases that customize the bits of behavours per model trype
# Have a look at design patterns (frameworks) for Python




