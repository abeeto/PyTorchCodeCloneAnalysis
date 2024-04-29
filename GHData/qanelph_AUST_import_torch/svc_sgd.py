import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model

import os
X = [] 
y = []

pth = 'fft/alg/'

for fname in os.listdir(pth):
    a = np.load(pth + fname).flatten()
    r = np.array([np.real(a), np.imag(a)]).flatten()
    X.append(r)
    y.append(0)

pth = 'fft/no/'
for fname in os.listdir(pth):
    a = np.load(pth + fname).flatten()
    r = np.array([np.real(a), np.imag(a)]).flatten()
    X.append(r)
    y.append(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (linear_model.SGDClassifier(max_iter=1000, tol=1e-3),)
#          svm.LinearSVC(C=C, max_iter=10000),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X_train, y_train) for clf in models)
scores = (clf.score(X_test, y_test) for clf in models)
print(list( scores ))
