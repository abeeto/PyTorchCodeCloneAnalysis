# * coding:utf-8 *
# @author    :mashagua
# @time      :2019/4/28 20:02
# @File      :lesson_1.py
# @Software  :PyCharm
# import torch
# x=torch.tensor([5.5,3])
# x=torch.zeros(5,3)
# print(x)
# y=torch.rand(5,3)
# print(y)
# x=torch.randn(4,4)
# y=x.view(16)
# print(y)
# a=torch.ones(6)
# b=a.numpy()
# print(b)
# #a,b共享内存
# print()
# #可以自动找GPU
# if torch.cuda.is_available():
#     device=torch.device("cuda")
#     y=torch.ones_like(x,device=device)
#     x=x.to(device)
#     z=x+y
#     print(z)
#     print(z.to("cpu",torch.double))
import torch
import numpy as np
"""
N:输入样本个数
D_in:每个样本维度
h:隐含层维度
D_out:输出样本维度
"""
N, D_in, h, D_out = 64, 1000, 100, 10
# # 产生一些随机样本
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
# w1 = np.random.randn(D_in, h)
# w2 = np.random.randn(h, D_out)
# lr = 1e-6
# for it in range(5000):
#     # forward
#     h = x.dot(w1)  # N*H
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)
#     # compute loss
#     loss = np.square(y_pred - y).sum()
#     print(it, loss)
#     # backward compute gradient
#     grad_y_pred = 2 * (y_pred - y)
#     # 本来应该是grad_y_pred-y,为了方便把常数项去掉了
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)
#     # update w1 and w2
#     w1 -= lr * w1
#     w2 -= lr * w2
# print("*" * 50)
# torch的对应的方法
# x = torch.randn(N,D_in)
# #创建cuda的时候可以用x=torch.randn(N,D_in).to("cuda:0")
# y = torch.randn(N, D_out)
# w1 = torch.randn(D_in, h)
# w2 = torch.randn(h, D_out)
# lr = 1e-6
# for it in range(500):
#     # forward
#     h = x.mm(w1)  # N*H
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)
#     # compute loss
#     loss = (y_pred - y).pow(2).sum().item()
#     print(it, loss)
#     # backward compute gradient
#     grad_y_pred = 2 * (y_pred - y)
#     # 本来应该是grad_y_pred-y,为了方便把常数项去掉了
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)
#     # update w1 and w2
#     w1 -= lr * w1
#     w2 -= lr * w2
# print("*"*100)
#
# x=torch.tensor(1.,requires_grad=True)
# w=torch.tensor(1.,requires_grad=True)
# b=torch.tensor(1.,requires_grad=True)
# y=w*x+b
# y.backward()
# print(w.grad)

# x = torch.randn(N,D_in)
# #创建cuda的时候可以用x=torch.randn(N,D_in).to("cuda:0")
# y = torch.randn(N, D_out)
# w1 = torch.randn(D_in, h,requires_grad=True)
# w2 = torch.randn(h, D_out,requires_grad=True)
# lr = 1e-6
# for it in range(500):
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)  # N*H
#     # compute loss
#     loss = (y_pred - y).pow(2).sum()
#     print(it,loss.item())
#     # backward compute gradient
#     loss.backward()
#     with torch.no_grad():
#         w1-=lr*w1.grad
#         w2-=lr*w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()


# x = torch.randn(N,D_in)
# #创建cuda的时候可以用x=torch.randn(N,D_in).to("cuda:0")
# y = torch.randn(N, D_out)
# model=torch.nn.Sequential(
#     torch.nn.Linear(D_in,h),#线性层
#     torch.nn.ReLU(),#relu层
#     torch.nn.Linear(h,D_out),#线性层
# )
# torch.nn.init.normal(model[0].weight)
# torch.nn.init.normal(model[2].weight)
#
# #model=model.cuda()
# lr=1e-6
# loss_fn=torch.nn.MSELoss(reduction="sum")
# for it in range(500):
#     y_pred = model(x)
#     # compute loss
#     loss = loss_fn(y_pred,y)
#     print(it,loss.item())
#     # backward compute gradient
#     loss.backward()
#     with torch.no_grad():
#         for param in model.parameters():
#             param-=lr*param.grad
#     model.zero_grad()


# x = torch.randn(N,D_in)
# #创建cuda的时候可以用x=torch.randn(N,D_in).to("cuda:0")
# y = torch.randn(N, D_out)
# model=torch.nn.Sequential(
#     torch.nn.Linear(D_in,h),#线性层
#     torch.nn.ReLU(),#relu层
#     torch.nn.Linear(h,D_out),#线性层
# )
# torch.nn.init.normal(model[0].weight)
# torch.nn.init.normal(model[2].weight)

# model=model.cuda()
# lr=1e-6
# loss_fn=torch.nn.MSELoss(reduction="sum")
# optimizer=torch.optim.Adam(model.parameters(),lr=lr)
# for it in range(500):
#     y_pred = model(x)
#     # compute loss
#     loss = loss_fn(y_pred,y)
#     print(it,loss.item())
#     # backward compute gradient
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


x = torch.randn(N, D_in)
# 创建cuda的时候可以用x=torch.randn(N,D_in).to("cuda:0")
y = torch.randn(N, D_out)
lr=1e-6

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, h, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, h, bias=False)
        self.linear2 = torch.nn.Linear(h, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TwoLayerNet('D_in', 'h', 'D_out')
loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for it in range(500):
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    print(it, loss.item())
    # backward compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
