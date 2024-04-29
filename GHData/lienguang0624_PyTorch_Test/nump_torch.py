import torch
import numpy as np
from torch.autograd import Variable
np_data = np.arange(6).reshape(2,3)
# numpy转换为tensor
torch_data = torch.from_numpy(np_data)
print(
    '\nnumpy转换为tensor',
    '\nnumpy:\n',np_data,
    '\ntorch_data:\n',torch_data
)
# tensor转换为numpy
tensor2numpy = torch_data.numpy()
print(
    '\ntensor转换为numpy',
    '\ntensor2numpy:\n',tensor2numpy
)

# 类型转换
data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)
print(
    '\n类型转换',
    '\ntensor:\n',tensor
)

# 绝对值
data = [[-1,-2],[1,2]]
tensor = torch.FloatTensor(data)
print(
    '\n绝对值',
    '\ntensor:\n',np.abs(data),
    '\ntensor:\n',torch.abs(tensor)
)

#矩阵相乘
print(
    '\n矩阵相乘',
    '\nnumpy:\n',np.matmul(data,data),
    '\ntorch:\n',torch.mm(tensor,tensor)
)
#设置变量
data = [[-1,-2],[1,2]]
tensor = torch.FloatTensor(data)
variable = Variable(tensor,requires_grad = True)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(
    't_out:\n',t_out,
    '\nv_out:\n',v_out
)
v_out.backward()
print('求取梯度：\nvariable.grad:\n',variable.grad)
