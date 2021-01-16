from dataset_reader import DataReader
import numpy as np
from sklearn.model_selection import train_test_split
from factory import Factory

if __name__ == '__main__':
    dataset = DataReader("./dataset/magic04.data")
    dataset_extract = dataset.read()
    samples, labels = np.array(dataset_extract[0]), np.array(dataset_extract[1])
    samples_train, samples_test, labels_train, labels_test = train_test_split(samples, labels, test_size=0.3, random_state=None)
    factory = Factory()
    knn = factory.factory("k_nearest_neighbor", samples_train, samples_test, labels_train, labels_test)
    print(knn.__class__.__name__)