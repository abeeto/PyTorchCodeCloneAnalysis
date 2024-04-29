import torch
from torch.nn import functional as F

# y = ax + b  =>  ln(y) = ln(ax + b)
# y = ax + b  =>  e^y = e^(ax + b)
# σ = sigmoid(z) = 1 / (1 + e^(-z))
# σ = sigmoid(z) = 1 / (1 + e^(-Xw))
# 一般的激活阈值为0.5。

# 几率的形式表现σ，即"σ / (1 - σ)"。称为对数几率回归logistic reg↓
#
# ln(σ / (1 - σ)) = ln((1 / (1 + e^(-Xw))) / (1 - (1 / (1 + e^(-Xw)))))
#                 = ln((1 / (1 + e^(-Xw))) / (e^(-Xw) / 1 + e^(-Xw)))
#                 = ln(1 / e^(-Xw))
#                 = ln(e^(Xw))
#                 = Xw
# σ可以理解为样本结果被预测为1的概率，那么1-σ即被预测为0的概率。

X = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)
# 与门
and_gate = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

w = torch.tensor([-0.2, 0.15, 0.15], dtype=torch.float32)


def logistic_reg(X, w):
    zhat = torch.mv(X, w)
    sigma = 1 / (1 + torch.exp(-zhat))
    # sigma = torch.sigmoid(zhat)
    # 设置阈值
    and_hat = torch.tensor([int(i) for i in sigma >= 0.5], dtype=torch.float32)
    return sigma, and_hat


sigma, and_hat = logistic_reg(X, w)
print(sigma)
print(and_hat)

# 其他常见二分类转换函数
# ReLU,σ = {
#   z,  (z > 0)
#   0,  (z <= 0)
# }
# 本质上就是 MAX(z, 0)

# 双曲正切函数 tanh = σ = (e^(2z) - 1) / (e^(2z) + 1)
# 双曲正切求导 tanh'(z) = 1 - tanh^2(z)

# sign 信号函数，1 if z > 0 else 0
# ------------------------------------------------------------------------
X1 = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
torch.random.manual_seed(996)
dense = torch.nn.Linear(2, 1)
zhat = dense(X1)
sigma = torch.sigmoid(zhat)  # torch.sign / F.relu / torch.tanh
y = [int(x) for x in sigma >= 0.5]
print(y)
