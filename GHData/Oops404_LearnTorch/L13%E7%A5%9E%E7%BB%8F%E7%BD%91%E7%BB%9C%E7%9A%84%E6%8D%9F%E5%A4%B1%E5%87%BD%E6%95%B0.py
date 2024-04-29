import torch
from torch.nn import MSELoss
import torch.nn as nn

# 回顾↓：
# 激活函数 sign sigmoid relu tanh 等。
# 多分类表现层 softmax。
"""
损失函数和机器学习中的类似，用来衡量真实值与预测值之间的差异，例如SSE。
通常用L(w)表示。
将损失函数L(w)转变为凸函数的数学方法，常见的有拉格朗日变换等。
在凸函数上求解L(w)的最小值对应的w方法，也就是以梯度下降为代表的优化算法。
"""

# 模型训练
# 定义基本模型 -> 定义损失函数 -> 定义优化算法 -> 以最小化损失函数为目标，求解权重
# --------------------------------------------------------------
# SSE = Σ(i = 1 ~ m) (z_i-zhat_i)^2
# MSE = (1/m) (Σ(i = 1 ~ m) (z_i-zhat_i)^2)

yhat = torch.randn(size=(50,), dtype=torch.float32)
y = torch.randn(size=(50,), dtype=torch.float32)

# 评估指标，reduction默认为mean，  mean 为MSE。 sum 输出sse
criterion = MSELoss()  # reduction = 'mean' / 'sum'
loss = criterion(yhat, y)
print(loss)
# --------------------------------------------------------------
# 二分类交叉熵损失函数
# L(w) = - Σ(i = 1 ~ m)(y_i * ln(σ_i) + (1 - y_i) * ln(1 - σ_i))
"""
极大似然估计MLE 回顾：
假设一个事件发生的概率很大，那么它应该经常发生，如果希望一件事情尽可能的
发生，应该尽量去增加这件事发生的概率。

为了事件A尽可能的发生，那我们只要寻找令其发生概率最大化的权重w。寻找相应
的权重w，使得目标事件发生的概率最大，就是极大似然估计的基本方法。
步骤：
构建似然函数P(w)，对其取对数，ln(P(w))。
在ln(P(w))上对权重w求导，并使导数为0，对权重进行求解。

二分类问题中，假定网络结果服从伯努利分布（01分布），样本i在由特征向量
x_i和权重向量w组成的预测函数中，样本标签被预测为1的概率为：
    P_1 = P(yhat_i = 1 | x_i, w) = σ => sigmoid()
    P_0 = P(yhat_i = 0 | x_i, w) = 1 - σ
    
我们期望得到完美情况，单个样本真实标签为1时的概率P_1 = 1,真实标签为0
的时候P_0 = 1，及百分百分对。
那么同时表示P_1和P_0，即逻辑回归的假设函数：
    P(yhat_i | x_i, w) = P_1^y_i * P_0^(1 - y_i)
为了损失函数最小，我们即是在期望 P(yhat_i | x_i, w) = 1。

那么多个样本上的情况，我们定义所有样本在特征张量X和权重向量w组成的预测函数
中，预测出所有可能的yhat的概率P为：
    P = Π(i = 1 ~ m) P(yhat_i | x_i, w)
      = Π(i = 1 ~ m) (P_1^y_i * P_0^(1 - y_i))
      = Π(i = 1 ~ m) (σ_i^y_i * (1 - σ_i)^(1 - y_i))
这就是逻辑回归的似然函数。对P取e为底的对数，再由log(A*B) = log(A) + log(B);
log(A^B) = B * log(A)，得逻辑回归的对数似然函数：
    ln(P) = ln(Π(i = 1 ~ m) (σ_i^y_i * (1 - σ_i)^(1 - y_i)))
          = Σ(i = 1 ~ m) ln(σ_i^y_i * (1 - σ_i)^(1 - y_i))
          = Σ(i = 1 ~ m) (ln(σ_i^y_i) + ln(1 - σ_i)^(1 - y_i))
          = Σ(i = 1 ~ m) (y_i * ln(σ_i) + (1 - y_i) * ln(1 - σ_i))
为了用来衡量损失，即取负：
    L(w) = -Σ(i = 1 ~ m) (y_i * ln(σ_i) + (1 - y_i) * ln(1 - σ_i))
那么问题就是转换成了求L(w)极小值。那么就是对它求导数为0咯，但是不好求啊，因此
需要一个优化算法。
"""
# 二分类交叉熵损失函数实现：
# sigma - sigmoid(z)
# z = Xw
# 假设有m个样本
m = 6 * pow(10, 6)  # 60000个
torch.random.manual_seed(996)
X = torch.rand((m, 4), dtype=torch.float32)
w = torch.rand((4, 1), dtype=torch.float32)
y = torch.randint(low=0, high=2, size=(m, 1), dtype=torch.float32)

zhat = torch.mm(X, w)
sigma = torch.sigmoid(zhat)
print(sigma.shape)
# 对tensor进行运算尽量使用torch中的函数，包括sum
# 计算每个的平均1/m
loss = -(1 / m) * torch.sum(y * torch.log(sigma) + (1 - y) * (torch.log(1 - sigma)))
print(loss)

# torch中实现的二分类交叉熵损失函数: BCEWithLogitsLoss 与 BCELoss。
criterion = nn.BCELoss()  # reduction=mean/sum/none ,默认mean, none即不求和返回每个的交叉熵计算结果的矩阵
loss = criterion(sigma, y)
print(loss)

criterion = nn.BCEWithLogitsLoss()
loss = criterion(zhat, y)
print(loss)
# --------------------------------------------------------------
"""
多酚类交叉熵，由二分类推到而来，对于多酚类问题，σ即softmax返回的结果。
    P_k = P(yhat_i = k | x_i, w) = σ
    
假设有三分类问题[1,2,3]，三类别概率为：
    P_1 = P(yhat_i = 1 | x_i, w) = σ_1
    P_2 = P(yhat_i = 2 | x_i, w) = σ_2
    P_3 = P(yhat_i = 3 | x_i, w) = σ_3
    
三分类的概率组合很难写出，因此我们将三分类onehot成二分类，(/≧▽≦)/
    1：（1，0，0），
    2：（0，1，0），
    3：（0，0，1），
假设一个样本有k个分类，三分类即k=3：
    P(yhat_i| x_i, w) = P_1^(y_i(k=1))*P_2^(y_i(k=2))*P_3^(y_i(k=3))
    递推：             = P_1^(y_i(k=1))*P_2^(y_i(k=2))*P_3^(y_i(k=3)) * ... * P_K^(y_i(k=K))
    由于onehot，(1,0,0)中0部分直接忽略，因此简化得：
    P(yhat_i| x_i, w) = P_j^(y_i(k=j))，j为真实标签内的编号。
    
期望概率最大：
    P = Π(i=1~m) P(yhat_i | x_i, w)
      = Π(i=1~m) P_j^(y_i(k=j))
      = Π(i=1~m) σ_j^(y_i(k=j))
和二分类一样求极大似然估计，求对数
    ln(P) = ln(Π(i=1~m) σ_j^(y_i(k=j)))
          = Σ(i=1~m) ln(σ_j^(y_i(k=j)))
          = Σ(i=1~m) y_i(k=j)ln(σ_j)
    L(w)  = -Σ(i=1~m) y_i(k=j)ln(σ_j)
    
在pytorch中，函数调用实现
    L(w)  = -Σ(i=1~m) y_i(k=j)ln(σ_j)
            ------------------
                  NLLLOSS
                              -------
                              LogSoftmax
    NLLLOSS：负对数似然 negative log likelihood function
"""
N = 6 * pow(10, 6)
X = torch.rand((m, 4), dtype=torch.float32)
w = torch.rand((4, 3), dtype=torch.float32)
y = torch.randint(low=0, high=3, size=(m,), dtype=torch.float32)

zhat = torch.mm(X, w)
sigma = torch.sigmoid(zhat)

logsoftmax = nn.LogSoftmax(dim=1)
logsigma = logsoftmax(zhat)

criterion = nn.NLLLoss()
loss = criterion(logsigma, y.long())
print(loss)

# 其他包装好的类 nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
print(criterion(logsigma, y.long()))
