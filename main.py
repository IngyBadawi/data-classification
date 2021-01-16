from dataset_reader import DataReader
import numpy as np
from k_nearest_neighbors import KNearestNeighbors
from ada_boost import AdaBoost
from random_forest import RandomForest
from naive_bayes import NaiveBayes
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset = DataReader("./dataset/magic04.data")
    dataset_extract = dataset.read()
    samples, labels = np.array(dataset_extract[0]), np.array(dataset_extract[1])
    samples_train, samples_test, labels_train, labels_test = train_test_split(samples, labels, test_size=0.3, random_state=None)
