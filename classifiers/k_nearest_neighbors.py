from classifiers.classifiers import Classifiers
from constants import *
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Classifiers):
    def __k_nearest_neighbor(self):
        n_neighbors = list(range(1, int(round(HADRONS_SIZE**0.5))))
        hyperparameters = dict(n_neighbors = n_neighbors)
        knn = KNeighborsClassifier()
        clf = GridSearchCV(knn, hyperparameters, cv=10)
        best_model = clf.fit(self.sample_train, self.label_train)
        best_params = best_model.best_params_
        best_estimator = best_model.best_estimator_
        cv_scores = best_model.cv_results_
        print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
        print("Best: %f using %s" % (best_model.best_score_, best_params))
        self.label_predicted = best_model.predict(self.sample_test)
        self.calculate_results()
        return n_neighbors, best_model.cv_results_['mean_test_score']

    def train(self):
        estimators, mean_scores = self.__k_nearest_neighbor()
        self.plot(estimators,mean_scores,'K-Nearest Neighbors','N-Neighbors', 'Fitting Scores')
