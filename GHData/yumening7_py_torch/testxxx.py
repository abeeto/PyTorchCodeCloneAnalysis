import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 创建一个数据源
x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

# 使用不同的激活函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
# softplus激活函数用于离散型的数据分类，不适用于连续性的数据
y_softplus = F.softplus(x).data.numpy()

# 绘图, num是图形窗口的id， figsize是尺寸，单位英尺
plt.figure(num=1, figsize=(8, 6))
# 绘制子图，前两个数分别表示行列，第三个数表示子图所在的位置
plt.subplot(221)
# 绘图
plt.plot(x_np, y_relu, c='red', label='relu')
# 设置y轴的刻度
plt.ylim((-1, 5))
# 设置图例位置
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='r', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='r', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='r', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()