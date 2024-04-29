import torch

t1 = torch.Tensor(1)
t2 = torch.tensor(1)

print("t1的值为{}，t1的数据类型为{}".format(t1, t1.type()))
print("t2的值为{}，t1的数据类型为{}".format(t2, t2.type()))
