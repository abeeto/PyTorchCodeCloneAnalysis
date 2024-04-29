"""
在这里使用pytorch的model，也就是nn
"""
import torch
from torch.autograd import Variable
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device('cpu')
# x = Variable(torch.randn(N, D_in), requires_grad=False)
# y = Variable(torch.ones(N, D_out), requires_grad=False)
# 直接这样初始化就可以了
x = torch.randn(N, D_in, device=device, dtype=torch.float, requires_grad=True)
y = torch.randn(N, D_out, device=device, dtype=torch.float, requires_grad=True)
model = nn.Sequential(  # 这个就相当于是一个计算步骤,把输入塞进去，获得输出
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, H),
    nn.ReLU(),
    # 接下来都是可以选取的集中激活函数
    # nn.Tanh(),
    # nn.Sigmoid(),
    nn.Linear(H, D_out),
)
# 这个是均方损失函数,获得一个函数
# 如果是False，返回是向量的和，如果是True，返回的是平均值
# MSE均方误差
# RMSE均方根误差，就是加一个开根号
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 5e-4
losses=[]
for t in range(20):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    # loss = (y_pred - y).pow(2).sum()
    print("第{}次迭代，loss值为：{}".format(t, loss))
    model.zero_grad()  # 这一步不能缺少,这一步把所有参数的梯度置为0，在这基础之上进行下一步运算
    loss.backward()
    losses.append(loss.item())
    with torch.no_grad():  # 必须有这一行，如果去掉data的话
        for param in model.parameters():
            # param.data -= learning_rate * param.grad.data
            param -= learning_rate * param.grad
print(losses)