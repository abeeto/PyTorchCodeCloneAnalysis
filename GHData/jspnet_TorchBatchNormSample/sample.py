import torch
import torch.nn as nn

# 点数定義             国  数   英  社  理
x_in = torch.tensor([[99, 10, 85, 90, 1],
                     [99, 25, 75, 90, 1],
                     [99, 30, 21, 90, 1],
                     [99, 15, 80, 90, 1],
                     [99, 12, 84, 91, 1],
                     [99, 20, 85, 91, 99]],
                    dtype=torch.float)

print("平均: ", torch.mean(x_in, 0))
print("分散: ", torch.var(x_in, 0))

# BatchNormalization 式の定義　※5科目
m = nn.BatchNorm1d(5)

print("-----")
print("BatchNormalization: ")
print(m(x_in))

print("-----")
print("BatchNormalization(1/10000): ")
x = x_in / 10000
print(m(x))

