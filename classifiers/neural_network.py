from classifiers.classifiers import Classifiers
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


class NN(BaseEstimator):
    def __init__(self, h1=10, h2=10):
        self.h1, self.h2 = h1, h2
        self.model = Sequential()
        self.model.add(Dense(self.h1, input_dim=10, activation='relu'))
        self.model.add(Dense(self.h2,  activation='relu'))
        self.model.add(Dense(2, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def fit(self, X, y):
        self.model.fit(X, y, epochs=100, batch_size=64, shuffle=True, verbose=0)

    def predict(self, X):
        return NeuralNetwork.onehot_decode(self.model.predict(X))

    def score(self, X, y):
        return accuracy_score(NeuralNetwork.onehot_decode(y), self.predict(X))


class NeuralNetwork(Classifiers):
    def __neural_network(self):
        onehot_train_labels = self.onehot_encode(self.label_train)
        # h1 = list(range(8, 11))
        # h2 = list(range(8, 11))
        h1 = [3, 4, 9 ,10] # After trial and error, the best values of h1 lie in this range, h1 is the number of hidden units in layer 1
        h2 = [3, 4, 9, 10] # After trial and error, the best values of h2 lie in this range, h2 is the number of hidden units in layer 2
        x = []
        for _h1 in h1:
            for _h2 in h2:
                st = str(_h1) + ','+ str(_h2)
                x.append(st)
        x = np.array(x)

        param_grid = {
            'h1' : h1,
            'h2' : h2
        }
        grid_search = GridSearchCV(NN(), param_grid=param_grid, cv=5)
        grid_search.fit(self.sample_train, onehot_train_labels)
        best_params = grid_search.best_params_
        results = grid_search.cv_results_
        print('Best params: ', best_params)
        best_estimator = grid_search.best_estimator_

        # Best values were h1=9 & h2=4
        #best_estimator = NN(h1=9, h2=4)

        best_estimator.fit(self.sample_train, onehot_train_labels)
        self.label_predicted = best_estimator.predict(self.sample_test)
        self.calculate_results()
        return x, results['mean_test_score']

    def train(self):
        hidden_layers, results = self.__neural_network()
        self.plot(hidden_layers, results, "Neural Network", "Hidden Layers h1,h2", "Fitting scores")

    @staticmethod
    def onehot_encode( old_labels):
        onehot_labels = []
        for label in old_labels:
            if label == 'g':
                onehot_labels.append([0, 1])
            else:
                onehot_labels.append([1, 0])
        return np.array(onehot_labels)

    @staticmethod
    def onehot_decode(old_labels):
        def_labels = []
        for onehot in old_labels:
            if onehot[0] > onehot[1]:
                def_labels.append('h')
            else:
                def_labels.append('g')
        return np.array(def_labels)
