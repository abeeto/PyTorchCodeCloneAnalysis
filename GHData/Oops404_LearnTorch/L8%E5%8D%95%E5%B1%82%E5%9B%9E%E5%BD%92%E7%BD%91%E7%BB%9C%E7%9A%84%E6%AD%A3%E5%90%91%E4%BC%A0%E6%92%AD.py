from torch.nn import Linear
import torch

# data = [
#     [1, 0, 0, -0.2],
#     [1, 1, 0, -0.05],
#     [1, 0, 1, -0.05],
#     [1, 1, 1, 0.1]
# ]

X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)

# 上一层往这一层的神经元传输数据的个数为2，这一层（在这里也就是输出层）的接受数据的神经元个数为1
# 设置随机数种子可以控制权重和截距，即伪随机。 torch.random.manual_seed() 人为设置随机数种子
# 输入层由特征矩阵决定了，所以不需要显式定义
torch.random.manual_seed(996)
output = Linear(2, 1)  # bias = False

# 自动生成的权重w
print(output.weight)
# 自动生成的截距c
print(output.bias)

zhat = output(X)

print(zhat)
