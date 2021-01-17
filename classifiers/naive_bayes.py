from constants import *
from classifiers.classifiers import Classifiers
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(Classifiers):

    def __naive_bayes(self):
        gnb = GaussianNB()
        self.label_predicted = gnb.fit(self.sample_train, self.label_train).predict(self.sample_test)
        self.calculate_results()

    def train(self):
        self.__naive_bayes()
