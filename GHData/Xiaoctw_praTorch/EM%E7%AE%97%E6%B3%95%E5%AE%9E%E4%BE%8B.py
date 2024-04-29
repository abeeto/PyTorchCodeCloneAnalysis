# coding:UTF-8
'''
Created on 2015年6月7日
@author: zhaozhiyong
'''
from __future__ import division
from numpy import *
import math as mt


def main():
    # 首先生成一些用于测试的样本
    # 指定两个高斯分布的参数，这两个高斯分布的方差相同
    sigma = 6
    miu_1 = 2
    miu_2 = 10

    # 随机均匀选择两个高斯分布，用于生成样本值
    N = 1000
    X = zeros((1, N))
    for i in range(N):
        if random.random() > 0.5:  # 使用的是numpy模块中的random
            X[0, i] = random.randn() * sigma + miu_1
        else:
            X[0, i] = random.randn() * sigma + miu_2

    # 上述步骤已经生成样本
    # 对生成的样本，使用EM算法计算其均值miu

    # 取miu的初始值
    k = 2
    miu = random.random((1, k))
    # miu = mat([40.0, 20.0])
    Expectations = zeros((N, k))

    for step in range(1000):  # 设置迭代次数
        # 步骤1，计算期望
        for i in range(N):
            # 计算分母
            denominator = 0
            for j in range(k):
                denominator = denominator + mt.exp(-1 / (2 * sigma ** 2) * (X[0, i] - miu[0, j]) ** 2)

            # 计算分子
            for j in range(k):
                numerator = mt.exp(-1 / (2 * sigma ** 2) * (X[0, i] - miu[0, j]) ** 2)
                Expectations[i, j] = numerator / denominator

        # 步骤2，求期望的最大
        # oldMiu = miu
        oldMiu = zeros((1, k))
        for j in range(k):
            oldMiu[0, j] = miu[0, j]
            numerator = 0
            denominator = 0
            for i in range(N):
                numerator = numerator + Expectations[i, j] * X[0, i]
                denominator = denominator + Expectations[i, j]
            miu[0, j] = numerator / denominator

        # 判断是否满足要求
        epsilon = 0.0001
        if sum(abs(miu - oldMiu)) < epsilon:
            break

        print(step)
        print(miu)

if __name__ == '__main__':
    main()




