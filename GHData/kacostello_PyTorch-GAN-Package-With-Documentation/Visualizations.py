# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 10:53:16 2021

@author: Kyle Costello
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def runGan():
    data = pd.read_csv(
        r'C:\Users\Kyle Costello\Downloads\pytorch_GAN_Package-main\pytorch_GAN_Package-main\winequality-white.csv',
        sep=";")
    print(np.shape(data))
    print(data)
    data.index.name = 'record_id'
    print(data)
    data.plot(kind='hist', x='fixed acidity', y='volatile acidity')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='citric acid')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='residual sugar')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='chlorides')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='free sulfur dioxide')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='total sulfur dioxide')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='density')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='pH')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='sulphates')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='alcohol')
    plt.show()

    data.plot(kind='hist', x='fixed acidity', y='quality')
    plt.show()

    data = data.to_numpy()
    print(data)
    print(data[0])

    # gan = SimpleGANTrainer.SimpleGANTrainer()
    # gan.train()


if __name__ == "__main__":
    runGan()
