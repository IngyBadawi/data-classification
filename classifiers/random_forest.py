from sklearn.ensemble import RandomForestClassifier

from constants import *
from classifiers.classifiers import Classifiers


class RandomForest(Classifiers):
    def __random_forest(self):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(self.sample_train, self.label_train)
        self.label_predicted = clf.predict(self.sample_test)
        self.calculate_results()

    def train(self):
        self.__random_forest()
