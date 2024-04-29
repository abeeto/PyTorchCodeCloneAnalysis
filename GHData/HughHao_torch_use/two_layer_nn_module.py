# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:43
# @Author : hhq
# @File : two_layer_nn_module.py
import torch


class TwoLayerNet(torch.nn.Module):  # 定义一个继承了torch.nn.Module的子类，父类是torch.nn.Module
    def __init__(self, D_in, H, D_out):
        # In the constructor we instantiate two nn.Linear modules and assign them as\n",
        # member variables.\n",
        super(TwoLayerNet, self).__init__()  # 继承别的类意味着使用super
        self.linear1 = torch.nn.Linear(D_in, H)  # 定义第一层线性连接
        self.linear2 = torch.nn.Linear(H, D_out)  # 第二层连接

    def forward(self, x):
        # In the forward function we accept a Tensor of input data and we must return\n",
        # a Tensor of output data. We can use Modules defined in the constructor as\n",
        # well as arbitrary operators on Tensors.\n",
        h_relu = self.linear1(x).clamp(min=0)  # relu激活函数
        y_pred = self.linear2(h_relu)  # 第二层的输入为第一层的输出
        return y_pred  # 第二层输出预测值 长度为D_out
        # N is batch size; D_in is input dimension",
        # H is hidden dimension; D_out is output dimension.\n",


N, D_in, H, D_out = 64, 1000, 100, 10
"# Create random Tensors to hold inputs and outputs\n"
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# Construct our model by instantiating the class defined above\n",
model = TwoLayerNet(D_in, H, D_out)  # 初始化神经网络模型
# Construct our loss function and an Optimizer. The call to model.parameters()\n",
# in the SGD constructor will contain the learnable parameters of the two\n",
# nn.Linear modules which are members of the model.\n",
criterion = torch.nn.MSELoss(reduction='sum')  # 定义最小化损失方差函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)  # 初始化优化器
loss_be = 100000000
while True:  # 梯度下降500次，也可以设置成损失小到一定程度即停止训练
    # Forward pass: Compute predicted y by passing x to the model\n",
    # y_pred = model.forward(x)
    y_pred = model(x)  # 此行和上一行注释掉的作用相同
    # Compute and print loss\n",
    loss = criterion(y_pred, y)  # 计算损失
    # print(loss)
    print(loss.item())  # 输出损失
    # Zero gradients, perform a backward pass, and update the weights.\n",
    optimizer.zero_grad()  # 优化器梯度下降,更新模型参数,即更新模型
    loss.backward()  # 反向传播
    optimizer.step()  # 迭代
    if loss_be > loss:
        loss_be = loss
    else:
        break
