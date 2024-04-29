import matplotlib
#matplotlib.rcParams['backend'] = 'Qt4Agg'
matplotlib.rcParams['backend'] = 'Agg'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import csv
import math


def norm(f):
    tot = 0
    for x in f:
        tot = tot + x * x
    return math.sqrt(tot)


def dot(f1, f2):
    nf1 = norm(f1)
    nf2 = norm(f2)
    tot = 0
    for (x, y) in zip(f1, f2):
        tot = tot + x * y
    return (tot) / (nf1 * nf2)


def extract_name_number(fn):
    base = fn.split('/')[-1]
    number = base.split('_')[-1]
    number = number.replace('.jpg', '')
    name = '_'.join(base.split('_')[0:-1])
    return (name, number)


def load_feature_file(filename):
    f = open(filename)
    l = f.readlines()
    l = l[1:]
    d = {}
    for item_ in l:
        item = item_.strip().split(',')
        (fn, number) = extract_name_number(item[0])
        feat = [float(x) for x in item_.strip().split(',')[1:513]]
        d[(fn, number)] = feat
    return d


def main():
    d = load_feature_file('lfwall_resnet18_deep_features.csv')
    f = open('pairs.txt')
    l = f.readlines()
    l = l[1:]
    cnt = 0
    res = []
    labels = []
    for i in range(10):
        for j in range(300):
            item = l[cnt]
            item = item.strip().split('\t')
            f1 = d[(item[0], item[1].zfill(4))]
            f2 = d[(item[0], item[2].zfill(4))]
            res.append(dot(f1, f2))
            labels.append(1)

            cnt = cnt + 1

        for j in range(300):
            item = l[cnt]
            item = item.strip().split('\t')
            f1 = d[(item[0], item[1].zfill(4))]
            f2 = d[(item[2], item[3].zfill(4))]
            res.append(dot(f1, f2))
            labels.append(-1)

            cnt = cnt + 1

    fpr, tpr, _ = roc_curve(np.array(labels), np.array(res))
    # roc_auc = auc(fpr, tpr)
    plt.semilogx(fpr, tpr, linewidth=2, label='resnet18')
    mn = 10
    eer = 0
    for i in range(fpr.shape[0]):
        v = abs((1 - fpr[i]) - tpr[i])
        if v < mn:
            mn = v
            eer = tpr[i]
    print eer
    plt.legend(loc='lower right')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.grid(True)
    plt.savefig('roc.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    main()
