from classifiers.k_nearest_neighbors import KNearestNeighbors
from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from classifiers.ada_boost import AdaBoost
from classifiers.naive_bayes import NaiveBayes


class Factory():
    def __init__(self):
        self.classifiers = {
            "k_nearest_neighbor": KNearestNeighbors,
            "decision_tree": DecisionTree,
            "random_forest": RandomForest,
            "ada_boost": AdaBoost,
            "naive_bayes": NaiveBayes
        }

    def factory(self, classifier, samples_train, samples_test, labels_train, labels_test):
        return self.classifiers[classifier](samples_train, samples_test, labels_train, labels_test)