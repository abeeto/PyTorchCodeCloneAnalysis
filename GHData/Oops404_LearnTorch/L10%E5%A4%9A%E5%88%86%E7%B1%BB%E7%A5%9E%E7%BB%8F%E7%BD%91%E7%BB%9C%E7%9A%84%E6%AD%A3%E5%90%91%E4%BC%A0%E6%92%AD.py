import torch
import numpy as np


# softmax = e^(z[k]) / Σ(0~k){e^z}

# 案例 有 苹果、梨、百香果，样本i被分类为百香果的概率 σ(百香果)=：
# σ(百香果) = e^(z[百香果]) / e^(z[苹果]) + e^(z[梨]) + e^(z[百香果])

# 可以写为
def softmax(z1):
    # 存在一个广播，细品
    return torch.exp(z1) / torch.sum(torch.exp(z1))


# 优化办法
# 定义softmax函数
# def softmax(z):
#     c = np.max(z)
#     exp_z = np.exp(z - c)  # 溢出对策
#     sum_exp_z = np.sum(exp_z)
#     o = exp_z / sum_exp_z
#     return o


# e^1000 指数级别，必然直接崩
Z = torch.tensor([1010, 300, 990], dtype=torch.float32)
print(softmax(Z))

# 因此常用调整后的torch.softmax(z,0)
# 第二个参数为要进行softmax的那个维度的索引,即tensor.shape的索引
# 对于1维张量的索引，必然只有0， 其他可以有正负值，即正向还是反向
print(torch.softmax(Z, dim=0))

Z = torch.tensor([10, 9, 5], dtype=torch.float32)
print(softmax(Z))
print(torch.softmax(Z, 0))
# --------------------------------------------------------------------

X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
torch.random.manual_seed(996)
# 此时输出层上的神经元个数为3
dense = torch.nn.Linear(2, 3)
zhat = dense(X)
sigma = torch.softmax(zhat, 1)
print(sigma)
# 这里应当返回的正确结构是(4,3)，4个样本，每个样本都有自己的3个类别对应的3个概率↓
print(sigma.shape)
