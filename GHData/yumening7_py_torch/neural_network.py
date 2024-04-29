'''
@author: Dzh
@date: 2019/12/25 10:45
@file: neural_network.py
'''

import numpy as np
import matplotlib.pyplot as plt

'''
pylot使用rc配置文件来自定义图形的各种默认属性，称之为rc配置或rc参数。通过rc参数可以修改默认的属性，
包括窗体大小、每英寸的点数、线条宽度、颜色、样式、坐标轴、坐标和网络属性、文本、字体等。
rc参数存储在字典变量中，通过字典的方式进行访问
'''

# 设置figure_size尺寸
plt.rcParams['figure.figsize'] = (10.0, 8.0)
''' 
最近邻差值: 像素为正方形
Interpolation/resampling即插值，是一种图像处理方法，它可以为数码图像增加或减少象素的数目。
某些数码相机运用插值的方法创造出象素比传感器实际能产生象素多的图像，或创造数码变焦产生的图像。
实际上，几乎所有的图像处理软件支持一种或以上插值方法。
图像放大后锯齿现象的强弱直接反映了图像处理器插值运算的成熟程度
'''
plt.rcParams['image.interpolation'] = 'nearest'
# 设置 颜色 style为灰度输出，而不是彩色输出
plt.rcParams['image.cmap'] = 'gray'

# 得到一组下标为0的随机数
np.random.seed(0)
N = 100
D = 2
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')

for j in range(K):
    '''
    ix是用来迭代的，第一次的值是range(0, 100)，对应的下标为[0, 99]
    第二次的值是range(100，200)，对应的下标为[100，199]
    第三次的值是range(200, 300)，对应的下标为[200, 299]
    '''
    ix = range(N * j, N * (j + 1))
    # 每次生成从0到1的100个数据
    r = np.linspace(0.0, 1, N)
    # 第一次生成从 0 到 4 的100个数据，第二次是从 4 到8，第三次是从 8 到12
    t = np.linspace(j*4, (j+1)*4, N)
    # np.c_是拼接两列数组，r的取值范围是0到1，np.sin(t)和np.cos(t)的取值范围是-1到1
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    # 给300个点分类，每一百为一类，即[0，99]为0类，[100，199]为1类，[200,299]为2类
    y[ix] = j
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()

