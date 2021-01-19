from classifiers.k_nearest_neighbors import KNearestNeighbors
from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from classifiers.ada_boost import AdaBoost
from classifiers.naive_bayes import NaiveBayes
from classifiers.neural_network import NeuralNetwork


class Factory:
    def __init__(self):
        self.classifiers = {
            "k nearest neighbor": KNearestNeighbors,
            "decision tree": DecisionTree,
            "random forest": RandomForest,
            "ada boost": AdaBoost,
            "naive bayes": NaiveBayes,
            "neural network": NeuralNetwork
        }

    def get(self, classifier, samples, labels):
        return self.classifiers[classifier](samples, labels)
