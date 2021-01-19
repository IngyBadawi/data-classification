from classifiers.classifiers import Classifiers
from sklearn import tree
import matplotlib.pyplot as plt

class DecisionTree(Classifiers):
    def __decision_tree(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.sample_train, self.label_train)
        self.label_predicted = clf.predict(self.sample_test)
        print(f'tree score   {clf.score(self.sample_test, self.label_test)}')
        self.calculate_results()
        #tree.plot_tree(clf)
        #plt.show()

    def train(self):
        self.__decision_tree()
