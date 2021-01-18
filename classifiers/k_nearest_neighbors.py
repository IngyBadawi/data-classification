from classifiers.classifiers import Classifiers
from constants import *
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbors(Classifiers):
    def __k_nearest_neighbor(self):
        return
        n_neighbors = list(range(1, int(round(HADRONS_SIZE**0.5))))
        hyperparameters = dict(n_neighbors = n_neighbors)
        knn = KNeighborsClassifier()
        clf = GridSearchCV(knn, hyperparameters, cv=10)
        best_model = clf.fit(self.sample_train, self.label_train)
        print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
        self.label_predicted = best_model.predict(self.sample_test);
        self.calculate_results()

    def train(self):
        self.__k_nearest_neighbor()
