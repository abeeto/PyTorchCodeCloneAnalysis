from sklearn import svm
from sklearn.metrics import classification_report


class Classifier():
    def __init__(self, X_Train, X_Test, y_Train, y_test):
        self.X_Train = X_Train.cpu().numpy().squeeze(1)
        self.X_Test = X_Test.cpu().numpy().squeeze(1)
        self.y_Train = y_Train.flatten()
        self.y_test = y_test.flatten()
        self.build_classifier()

    def build_classifier(self):
        clf = svm.SVC(max_iter=-1)
        classifier = clf.fit(self.X_Train, self.y_Train)
        preds = classifier.predict(self.X_Test)
        print(classification_report(preds, self.y_test))
