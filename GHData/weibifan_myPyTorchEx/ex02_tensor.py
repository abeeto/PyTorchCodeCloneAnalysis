# -*- coding: utf-8 -*-
# weibifan 2022-10-3
# Tensor类及对象说明。需要显式将Tensor对象搬迁到GPU的Mem中。
# Tensor封装各种基本类型，比如int，float，string，矩阵等等。
# cuda 与 cpu的差别及使用方法

import torch
import numpy as np
"""
Tensor模型（Tensor对象）：类似Java的Object
3种基本状态，train, evalue, valide

Tensor对象的数据和代码，既可以在mem中，也可以在GPU上，需要显式的搬迁。

# 复杂数据对象：比如ndarray和Tensor
①能存储什么样的基本数据，比如字符串，矩阵等，
②支持那些操作，比如+-*/怎么定义，再比如矩阵乘法，字符串拼接等
③ 一般不能靠=完成对象的初始化。一般靠构造函数，专用函数！！！
④ndarray和Tensor可以桥接

https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/

"""

# 1）Python基本的矩阵。支持+-*。存于CPU的Mem
data = [[1, 2],[3, 4]]
# 2） Numpy的多维矩阵，支持广播机制。存于CPU的Mem
np_array = np.array(data)
# 3） PyTorch的Tensor，支持广播机制，默认存于CPU的Mem，通过函数可以存于GPU的Mem（需要to函数）
x_data = torch.tensor(data)

print(x_data.size())
# Tensor数据对象如何初始化，默认是在CPU上。
# 使用Numpy的多维矩阵，从另外一个Tensor对象。特殊是随机初始化。
tensor = torch.rand(3,4)

# 需要显式的搬迁到GPU上。数量量大时，很耗时。
if torch.cuda.is_available():
    tensor_gpu = tensor.to("cuda")

# Tensor的操作（CPU或者GPU），有100多个，包括+-*/, 包括矩阵乘法，线性变换，抽样等。2维tensor是关系表，支持投影等关系操作（slice等）。
tensor2= tensor * tensor # 在CPU上进行运算。

# 下面代码只能在GPU的机器上运行
# tensor2_gpu= tensor_gpu * tensor_gpu # 在GPU上进行运算。按元素相乘。
# tensor2_gpu = tensor_gpu.matmul(tensor_gpu.T)  # 在GPU上进行运算。矩阵乘法。

# 当Tensor对象在CPU中时，可以起一个别名，当做Numpy的多维数组，贡献数据。也就是支持Numpy多维数组的各种操作。
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)  #改变Tensor对象
print(f"t: {t}")
print(f"n: {n}")






