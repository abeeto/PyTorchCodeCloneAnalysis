# Tensor和NumPy相互转换

# 我们很容易用numpy()和from_numpy()将Tensor和NumPy中的数组相互转换。
# 但是需要注意的一点是： 这两个函数所产生的的Tensor和NumPy中的数组共享相同的内存（所以他们之间的转换很快），
# 改变其中一个时另一个也会改变！！！

import torch
'''
  Tensor转NumPy
  使用numpy()将Tensor转换成NumPy数组
'''
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)


'''
  NumPy数组转Tensor
  使用from_numpy()将NumPy数组转换成Tensor
'''
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)
# 所有在CPU上的Tensor（除了CharTensor）都支持与NumPy数组相互转换。

'''
  直接用torch.tensor()将NumPy数组转换成Tensor
'''

# 还有一个常用的将NumPy中的array转换成Tensor的方法就是torch.tensor(), 
# 需要注意的是，此方法总是会进行数据拷贝（就会消耗更多的时间和空间），所以返回的Tensor和原来的数据不再共享内存。
# 直接用torch.tensor()将NumPy数组转换成Tensor，需要注意的是该方法总是会进行数据拷贝，返回的Tensor和原来的数据不再共享内存。
c = torch.tensor(a)
a += 1
print(a, c)