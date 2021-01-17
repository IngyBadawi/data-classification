from constants import *
from dataset_reader import DataReader
from factory import Factory
import numpy as np

if __name__ == '__main__':
    dataset = DataReader(DATASET_PATH)
    dataset_extract = dataset.read()
    samples, labels = np.array(dataset_extract[0]), np.array(dataset_extract[1])
    factory = Factory()

    for classifier_name in CLASSIFIERS:
        classifier = factory.get(classifier_name, samples, labels)
        print(f'start training {classifier_name}')
        classifier.train()
