from constants import *
from classifiers.classifiers import Classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class AdaBoost(Classifiers):
    def __ada_boost(self):
        n_estimators = list(range(1, MAX_ESTIMATORS))
        hyperparameters = dict(n_estimators=n_estimators)
        abc = AdaBoostClassifier(learning_rate=1)
        grid_search = GridSearchCV(abc, hyperparameters, cv=5)
        gsf = grid_search.fit(self.sample_train, self.label_train)
        best_params = gsf.best_params_
        best_estimator = gsf.best_estimator_
        cv_scores = gsf.cv_results_
        print('Best estimator:', best_estimator.get_params()['n_estimators'])
        print("Best: %f using %s" % (gsf.best_score_, best_params))
        self.label_predicted = gsf.predict(self.sample_test)
        self.calculate_results()
        # means = gsf.cv_results_['mean_test_score']
        # stds = gsf.cv_results_['std_test_score']
        # params = gsf.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        return n_estimators, gsf.cv_results_['mean_test_score']

    def train(self):
        estimators, mean_scores = self.__ada_boost()
        self.plot(estimators, mean_scores,'AdaBoost','N-Estimators', 'Fitting Scores')
