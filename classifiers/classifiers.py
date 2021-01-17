from constants import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class Classifiers:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.sample_train, self.sample_test, self.label_train , self.label_test = train_test_split(samples, labels, test_size=0.3, random_state=0)
        self.label_predicted = None  # to be compared with label_test after testing sample_test

    def train(self):
        # each classifier has its own implementation
        pass

    def plot(self):
        pass

    def calculate_results(self):
        count = 0
        for i in range(len(self.label_test)):
            if self.label_predicted[i] != self.label_test[i]:
                count += 1
        print(f'Misplaced labels: {count}')
        print("Confusion Matrix: ", confusion_matrix(self.label_test, self.label_predicted))
        print(f'Accuracy : {accuracy_score(self.label_test, self.label_predicted) * 100} %')
        print("Report:",classification_report(self.label_test, self.label_predicted))