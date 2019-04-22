from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from filehandler import Filehandler


class Model:

    def __init__(self):
        self.enabled = False
        self.X_train = None
        self.y_train = None
        self.random_state = 20
        self.predictions = None
        self.base = {'model': None,
                    'stext': None,
                    'scores': None,
                    'cm': None}



    def set_dataset(self, folder, file):
        filehandler = Filehandler()
        self.X_train = filehandler.read_csv(folder, file + '_processed')
        self.y_train = filehandler.read_csv(folder, file + '_target')

    def fit(self):
        self.base['model'].fit(self.X_train, self.y_train)

    def score(self):
        self.base['scores'] = cross_val_score(self.base['model'], self.X_train, self.y_train, cv=10, scoring='roc_auc')
        print(self.base['scores'])
        return self.base['scores']

    def predict(self):
        self.predictions = cross_val_predict(self.base['model'], self.X_train, self.y_train, cv=10)

    def get_model_as_json(self):
        return self.base.__dict__.copy()

    def apply_pca(self):
        pca = PCA(n_components = 2)
        self.X_train = pca.fit_transform(self.X_train)
        explained_variance = pca.explained_variance_ratio_
        print(explained_variance)

class RandomForestClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.enabled = True
        self.base['stext'] = 'RFC'
        self.base['model'] = RandomForestClassifier(random_state=self.random_state)


class DecisionTreeClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.base['stext'] = 'DTC'
        self.base['model'] = DecisionTreeClassifier(random_state=self.random_state)


class AnnPerceptron(Model):
# Has only 1 neuron
# See perceptron overview here https://medium.com/cracking-the-data-science-interview/a-gentle-introduction-to-neural-networks-for-machine-learning-d5f3f8987786

class AnnMlpBinary(Model):
    def __init__(self):
        Model.__init__(self)
        self.enabled = False
        self.base['stext'] = 'ANNPCLF'
        self.base['model'] = KerasClassifier(build_fn=self.create_network, epochs=10, batch_size=100, verbose=0)

    def create_network(self):
        model = Sequential()
        model.add(Dense(64, input_dim=20, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  epochs=20,
                  batch_size=128)
        score = model.evaluate(x_test, y_test, batch_size=128)

class AnnFeedForward(Model):

   # Because this is a binary classification problem, one common choice is to use the sigmoid activation function in a one-unit output layer.

    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='rmsprop', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric


    # Train neural network
    history = network.fit(train_features,  # Features
                          train_target,  # Target vector
                          epochs=3,  # Number of epochs
                          verbose=1,  # Print description after each epoch
                          batch_size=100,  # Number of observations per batch
                          validation_data=(test_features, test_target))  # Data for evaluation

class ANNPerceptronClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.enabled = False
        self.base['stext'] = 'ANNPCLF'
        self.base['model'] = KerasClassifier(build_fn=self.create_network, epochs=10, batch_size=100, verbose=0)

    def create_network(self):
        network = Sequential()

        # Input layer with inputs matching 0 axis of tensor, hidden layer with 1 neuron
        network.add(Dense(output_dim=1, init='uniform', activation='relu', input_dim=self.X_train.shape[1]))

        # Output layer - sigmoid good for binary classification
        network.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

        # Binary cross entropy good for binary classification
        network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return network

    def set_dataset(self, folder, file):
        Model.set_dataset(self, folder, file)

    def fit(self):
        self.base['model'].fit(self.X_train, self.y_train)

    def predict(self):
        self.predictions = self.base['model'].predict(self.X_train)
        self.predictions = (self.predictions > 0.5)

