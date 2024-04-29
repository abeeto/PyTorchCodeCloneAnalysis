import torch
import numpy as np
import random
from matplotlib import pyplot as plt

# 创建数据集
true_w = [3.2, 4.5]
true_b = 2.7
num_examples = 1000
num_inputs = len(true_w)

# features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
features = torch.normal(0, 1, (num_examples, num_inputs), dtype=torch.float32)
# 0,0.1,training 结果很差，即使和noise相差100倍（0.1---0.001，也会差距很大）
labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, num_examples))  # noise 尽量小，和feature差别开来


# 读取小批量数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_mark = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, batch_mark), labels.index_select(0, batch_mark)


#
# #  test
# for f, l in data_iter(10, features, labels):
#     print(f)
#     print(l)
#     break
# 初始化参数
w = torch.normal(0, 0.1, (num_inputs, 1), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# print(w, b)


# 创建模型
def model(X, w, b):  # ---tensor,tensor,tensor
    return torch.mm(X, w) + b  # ----y


# 创建损失函数
def loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2  # --- tensor,loss,是w，b的函数


# 创建参数更新函数---调用该函数之前要。backward（）
def update(lr, params, batch_size):  # constant,w and b (tensor), constant
    for param in params:
        param.data -= lr * param.grad / batch_size


# trainning
epoch = 0
lr = 0.03
batch_size = 10
training_loss = 1
while training_loss > 5.2e-5:
    # 遍历data set：计算损失函数（是w，b的函数），bachward，更新params，grad清零。
    for x, y in data_iter(batch_size, features, labels):
        l = loss(model(x, w, b), y).sum()
        l.backward()
        update(lr, [w, b], batch_size)

        # w.grad.data.zero_()
        # b.grad.data.zero_()
        w.grad.zero_()
        b.grad.zero_()
    train_l = loss(model(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    training_loss = train_l.mean().item()
    epoch += 1

print(w)
print(b)
print(epoch + 1)
