from constants import *
from classifiers.classifiers import Classifiers
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost(Classifiers):
    def __ada_boost(self):
        abc = AdaBoostClassifier(n_estimators=100, learning_rate=1)
        model = abc.fit(self.sample_train, self.label_train)
        self.label_predicted = model.predict(self.sample_test)
        self.calculate_results()

    def train(self):
        self.__ada_boost()
