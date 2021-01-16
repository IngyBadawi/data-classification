from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class Classifiers:
    def __init__(self, sample_train, sample_test, label_train, label_test):
        self.sample_train = sample_train
        self.sample_test = sample_test
        self.label_train = label_train
        self.label_test = label_test
        self.label_predicted = None  # to be compared with label_test after testing sample_test

    def train(self):
        pass

    def calculate_results(self):
        print("Confusion Matrix: ", confusion_matrix(self.label_test, self.label_predicted))
        print(f'Accuracy : {accuracy_score(self.label_test, self.label_predicted) * 100}%')
        print("Report:",classification_report(self.label_test, self.label_predicted))