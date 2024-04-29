import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = torch.arange(1, 5).reshape(2, 2).float()

plt.plot(A[:, 0], A[:, 1], 'o')
plt.show()

# 三维空间上绘制SSE的图像
x = np.arange(-1, 3, 0.05)
y = np.arange(-1, 3, 0.05)

a, b = np.meshgrid(x, y)
SSE = (2 - a - b) ** 2 + (4 - 3 * a - b) ** 2

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(a, b, SSE, cmap='rainbow')
ax.contour(a, b, SSE, zdir='z', offset=0, cmap='rainbow')
plt.show()
# --------------------

'''
凹凸性
凸定义: 对于任意一个函数，如果有任意两个点满足👇，我们判定其为凸函数。
    （f(x1) + f(x2)）/ 2 >= f((x1 + x2) / 2)
'''
