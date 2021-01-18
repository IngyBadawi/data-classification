from constants import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

class Classifiers:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.sample_train, self.sample_test, self.label_train, self.label_test = train_test_split(samples, labels,
                                                                                                  test_size=0.3,
                                                                                                  random_state=0)
        self.label_predicted = None  # to be compared with label_test after testing sample_test
        self.saving_path = f'{OUTPUT_PATH}/{self.__class__.__name__}'

    def train(self):
        # each classifier has its own implementation
        pass

    def plot(self, n_estimators, mean_test_score, title='', xlabel='Estimators', ylabel='Scores'):
        plt.figure(figsize=(8, 8))
        # plt.plot(n_neighbors, best_model.cv_results_['mean_test_score'].tolist())
        plt.plot(n_estimators, mean_test_score)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


        #save data
        plt.savefig(self.saving_path+f'/{self.__class__.__name__}_scores_graph.png')
        plt.show()

    def calculate_results(self):
        misplaced_labels = 0
        for i in range(len(self.label_test)):
            if self.label_predicted[i] != self.label_test[i]:
                misplaced_labels += 1

        data = {
            'misplaced_labels' : misplaced_labels,
            'confusion_matrix' : confusion_matrix(self.label_test, self.label_predicted),
            'accuracy_score' : accuracy_score(self.label_test, self.label_predicted) * 100,
            'classification_report' : classification_report(self.label_test, self.label_predicted)
        }
        print(f"Misplaced labels: {data['misplaced_labels']}")
        print(f"Confusion Matrix: {data['confusion_matrix']}")
        print(f"Accuracy : {data['accuracy_score']} %")
        print(f"Report: {data['classification_report']}")

        #save data
        os.makedirs(self.saving_path, 0o666, True)
        with open(self.saving_path + "/data.txt", "w") as output:
            for key,val in data.items():
                output.write(f'{key}: \n{val}\n\n\n')
