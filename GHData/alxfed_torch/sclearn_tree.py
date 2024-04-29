# -*- coding: utf-8 -*-
"""...
"""
from sklearn import tree


if __name__ == '__main__':
    clf = tree.DecisionTreeClassifier()

    # [height, hair-length, voice-pitch]
    X = [[180, 15, 0],
         [167, 42, 1],
         [136, 35, 1],
         [174, 15, 0],
         [141, 28, 1]]

    Y = ['man', 'woman', 'woman', 'man', 'woman']

    clf = clf.fit(X, Y)
    prediction = clf.predict([[133, 37, 0]])
    print(prediction)
    print('\ndone')