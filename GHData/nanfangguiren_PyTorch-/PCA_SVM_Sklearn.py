"""
-*- coding: utf-8 -*-
@Time :  2022-04-02 15:15
@Author : nanfang
"""

import os
import struct
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def load_mnist(size, path, kind='train'):
    """
    加载MINIST数据集
    :param size: 加载数据集的大小
    :param path: 本地文件存放路径
    :param kind: train表示加载训练集，test表示加载测试集
    :return:
    """
    if kind == "test":
        kind = 't10k'
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images[:size], labels[:size]


def PCA_sklearn(train_data, test_data, k=0.9):
    pca = PCA(n_components=k)  # 选取维度个数
    # pca = PCA(n_components=0.99)    #主成分占了99%的方差比例
    pca.fit(train_data)  # 用X拟合模型。
    train_data_low = pca.transform(train_data)  # 对X进行降维，X被投影到之前从训练集模型中提取的第一个主成分上。
    test_data_low = pca.transform(test_data)

    return train_data_low, test_data_low


def SVM_sklearn(train_data, train_labels, test_data, test_labels):
    svm_model = svm.SVC()
    svm_model.fit(train_data, train_labels)
    train_score = svm_model.score(train_data, train_labels)
    print("训练集精度：{0}%".format(train_score*100))
    test_predict = svm_model.predict(test_data)
    accuracy = accuracy_score(y_true=test_labels,y_pred=test_predict)
    f1 = f1_score(y_true=test_labels,y_pred=test_predict, average='micro')
    print("测试集准确率：{0}%".format(accuracy*100))
    print("测试集f1_score：{0}%".format(f1*100))



if __name__ == '__main__':
    DATA_PATH = '../../../dataset/mnist/MNIST/raw'  # 数据在本地的地址
    train_data, train_labels = load_mnist(10000, DATA_PATH, kind='train')
    test_data, test_labels = load_mnist(200, DATA_PATH, kind='test')
    train_data, test_data = PCA_sklearn(train_data, test_data, k=0.9)
    SVM_sklearn(train_data, train_labels, test_data, test_labels)
