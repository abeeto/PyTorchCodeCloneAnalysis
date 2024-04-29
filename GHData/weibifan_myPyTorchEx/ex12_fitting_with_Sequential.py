# -*- coding: utf-8 -*-
# weibifan 2022-10-1
#  多项式拟合，使用nn.Sequential对象
# 第1种方法：使用Sequential定义模型
# ！！！ 假定1：所有模型默认都是batch方式工作，batch_size取决于模型和Mem
# ！！！ 假定2：所有模型默认都是多次使用训练数据，epoch是经验参数
'''

第1步：准备训练数据。包括两部分，输入数据和预期输出。

第2步：构建模型的对象

第3步：构建损失函数

第4步：训练，设置epoch
1）前向计算。将数据传递给模型对象。
2）计算损失。
3）初始化权重为0,
4）反向传播，计算梯度，
5）更新梯度。需要设置步长，不同函数不同步长。

第5步：获得权重


'''

import torch
import math
# Create Tensors to hold input and outputs.
# 准备训练数据。共2000个instance，
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
# 每个instance共3维
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
'''
第2步：构建模型：
第1层：线性映射 y= w1 * x **3 + w2 * x **2 + w3 * x，3个参数
第2层：
'''
# ！！！ 所有模型默认都是batch方式工作
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1), # 输入特征是3维，输出是1维，本质就是计算y=xw+b，其中x是3维

    # 将矩阵拼接成一个向量，默认，保留第0维，也就是instance个数，其他的进行拼接。
    # (0,1)表示行列都拼接
    torch.nn.Flatten(0, 1)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
# 第3步：构建损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')

# 第4步：模型训练
learning_rate = 1e-6  #改成1e-5都不行
for t in range(2000):  #epoch，这里的2000和数据集的2000没有关系

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    # 1）前向计算，前向传播
    y_pred = model(xx)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    # 2）计算损失
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    # 3）初始化权重为0,
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    # 4）反向传播，计算梯度。
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.

    # 使用梯度下降方法，更新梯度。
    with torch.no_grad(): #关闭自动的梯度更新，梯度还是要计算的。
        for param in model.parameters():   #获得参数。
            param -= learning_rate * param.grad # 更新梯度

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

b=linear_layer.bias.item()
w1=linear_layer.weight[:, 0].item()
w2=linear_layer.weight[:, 1].item()
w3=linear_layer.weight[:, 2].item()

# 绘制函数曲线  只在-pi和pi之间拟合很好，
import numpy as np
import math
import matplotlib.pyplot as plt
#sin & cos曲线

x = np.arange(-math.pi,math.pi, 0.1) #
y1 = np.sin(x)
y2 = np.cos(x)
y3 =  w3 * x**3 + w2 * x**2 + w1 * x +b
plt.plot(x,y1,label="sin")
plt.plot(x,y3,label="fit",linestyle = "--")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & fit')
plt.legend()   #打上标签
plt.show()

