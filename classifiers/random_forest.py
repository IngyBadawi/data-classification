from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from constants import *
from classifiers.classifiers import Classifiers


class RandomForest(Classifiers):
    def __random_forest(self):
        n_estimators = list(range(1, MAX_ESTIMATORS))
        hyperparameters = dict(n_estimators=n_estimators)
        clf = RandomForestClassifier()
        grid_search = GridSearchCV(clf, hyperparameters, cv=5)
        gsf = grid_search.fit(self.sample_train, self.label_train)
        best_params = gsf.best_params_
        best_estimator = gsf.best_estimator_
        cv_scores = gsf.cv_results_
        print('Best estimator:', best_estimator.get_params()['n_estimators'])
        print("Best: %f using %s" % (gsf.best_score_, best_params))
        self.label_predicted = gsf.predict(self.sample_test)
        self.calculate_results()
        return n_estimators, gsf.cv_results_['mean_test_score']

    def train(self):
        estimators, mean_scores = self.__random_forest()
        self.plot(estimators, mean_scores,'Random Forest','N-Estimators', 'Fitting Scores')
