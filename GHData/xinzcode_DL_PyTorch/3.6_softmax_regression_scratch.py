import torch
import torchvision
import numpy as np
from d2lzh_pytorch.utils import *

batch_size = 256
num_inputs = 784
num_outputs = 10
num_epochs, lr = 10, 0.1


# 获取和读取数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 初始化模型参数
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
# 同样需要模型参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
# 创建一个张量x，并设置其 requires_grad参数为True，程序将会追踪所有对于该张量的操作，
# 当完成计算后通过调用 .backward()，自动计算所有的梯度， 这个张量的所有梯度将会自动积累到 .grad 属性。

# 如何对多维Tensor按维度操作
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))  # 同一列（dim=0）求和 并在结果中保留行和列这两个维度（keepdim=True）。
# print(X.sum(dim=1, keepdim=True))  # 同一行（dim=1）求和

# softmax运算
# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))


# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)  # torch.mm 矩阵乘法






# 演示gather函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 变量y_hat是2个样本在3个类别的预测概率
y = torch.LongTensor([0, 2])   # y 标签
print(y.view(-1, 1))   # 等于print(y.view(2, 1))  先将所有元素排成一排，然后以2行2列的格式输出
y_hat.gather(1, y.view(-1, 1))  # dim=1 横向取值
print(y_hat.gather(1, y.view(-1, 1)))  # 以y为坐标按行取y_hat中的数 输出的分别是第一个0.1 和第三个0.5
# view 把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor。


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))





# 准确率函数
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
# print(accuracy(y_hat, y))


# 评价模型net在数据集data_iter上的准确率。
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


print(evaluate_accuracy(test_iter, net))




# 训练模型
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])



