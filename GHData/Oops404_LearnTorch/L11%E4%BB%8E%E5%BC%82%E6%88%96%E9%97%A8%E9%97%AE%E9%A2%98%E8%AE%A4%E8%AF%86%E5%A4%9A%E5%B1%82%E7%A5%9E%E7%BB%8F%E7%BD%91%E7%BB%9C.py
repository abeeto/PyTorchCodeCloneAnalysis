import torch


# 与门
def AND(X):
    w = torch.tensor([-0.2, 0.15, 0.15], dtype=torch.float32)
    zhat = torch.mv(X, w)
    andhat = torch.tensor([int(x) for x in zhat >= 0], dtype=torch.float32)
    return andhat


# 或门
def OR(X):
    w = torch.tensor([-0.08, 0.15, 0.15], dtype=torch.float32)  # 在这里我修改了b的数值
    zhat = torch.mv(X, w)
    yhat = torch.tensor([int(x) for x in zhat >= 0], dtype=torch.float32)
    return yhat


# 非与门
def NAND(X):
    w = torch.tensor([0.23, -0.15, -0.15], dtype=torch.float32)  # 和与门、或门都不同的权重
    zhat = torch.mv(X, w)
    yhat = torch.tensor([int(x) for x in zhat >= 0], dtype=torch.float32)
    return yhat


X = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]],
                 dtype=torch.float32)

x0 = torch.tensor([1, 1, 1, 1],
                  dtype=torch.float32)


# 异或门，相同为非
def XOR(X):
    # 输入值：
    input_1 = X
    # 中间层：
    sigma_nand = NAND(input_1)
    sigma_or = OR(input_1)
    x0 = torch.tensor([[1], [1], [1], [1]], dtype=torch.float32)
    # 输出层： cat合并，view均变成4行1列的二维形式
    input_2 = torch.cat((x0.view(4, 1), sigma_nand.view(4, 1), sigma_or.view(4, 1)), dim=1)
    y_and = AND(input_2)
    # print("NANE:", y_nand)
    # print("OR:", y_or)
    return y_and


print(XOR(X))
