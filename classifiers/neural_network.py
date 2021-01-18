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
    def __nerual_network(self):
        onehot_train_labels = self.onehot_encode(self.label_train)
        """
        param_grid = {
            'h1' : list(range(1, 10)),
            'h2' : list(range(1, 10))
        }
        grid_search = GridSearchCV(NN(), param_grid=param_grid, cv=5)
        grid_search.fit(self.sample_train, onehot_train_labels)
        best_params = grid_search.best_params_
        print('Best params: ', best_params)
        best_estimator = grid_search.best_estimator_
        """
        # Best values were h1=10 & h2=10
        best_estimator = NN(h1=10, h2=10)
        best_estimator.fit(self.sample_train, onehot_train_labels)
        self.label_predicted = best_estimator.predict(self.sample_test)


        self.calculate_results()
        # a = accuracy_score(self.onehot_decode(self.label_predicted), self.label_test)
        # print('Accuracy is:', a * 100)
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

    def train(self):
        self.__nerual_network()

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
