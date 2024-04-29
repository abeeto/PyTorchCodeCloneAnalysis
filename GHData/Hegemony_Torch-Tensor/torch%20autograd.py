import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

epoch_n = 20
# 训练次数为20次，通过循环的方式让程序进行20次训练
learning_rate = 1e-6
# 学习速率

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)
# Variable类对 Tensor 数据类型变量进行封装的操作在以上代码中还使用了
# requires_grad参数，这个参数的赋值类型是布尔型,如果requires_grad的值False，
# (默认为False)那么表示该变量在进行自动梯度计算的过程中不会保留梯度值
w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)
# 完成自动梯度需要用到 torch.autograd 包中的 Variable 类对我们定义的Tensor
# 数据类型变量进行封装，在封装后，计算图中的各个节点就是 Variable 对象，这样才
# 能应用自动梯度的功能。在选中了计算图中的某个节点时，这个节点
# 必定会是Variable对象 用X来代表我们选中的节点，那么X.data 代表 Tensor 数据
# 类型的变量，X.grad也是一个Variable对象，不过它表示的是X的梯度,在想访问梯度值
# 时需要使用 X.grad.data
# tensor.data和tensor.detach() 都是变量从图中分离，
# 但而这都是“原位操作 inplace operation”。
# （1）.data 是一个属性，而.detach()是一个方法；
# （2）.data 是不安全的，.detach()是安全的。

for epoch in range(epoch_n):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss.data))
    # print("Epoch:{},Loss:{:.4f}".format(epoch, loss.data.item()))

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

print('-' * 100)

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
y = x + 2
print(y)
z = y * y * 3
print(z)
out = z.mean()
print(out)
out.backward()
print(x.grad)
print('-' * 100)
