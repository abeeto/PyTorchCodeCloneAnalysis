import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)

a = [[1,1],[2,2],[3,3]]
b = torch.tensor(a)
print(a,'\n',b)

from torch.autograd import Variable # torch 中 Variable 模块

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)

print(variable)     #  Variable 形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy 形式
"""
[[ 1.  2.]
 [ 3.  4.]]
"""


t_out = torch.mean(tensor*tensor)       # x^2
# v_out = torch.mean(variable*variable)   # x^2
max_out = torch.max(variable*variable)
print(t_out)
# print(v_out)    # 7.5
print(max_out)

# v_out.backward()    # 模拟 v_out 的误差反向传递
max_out.backward()  

# v_out = 1/4 * sum(variable*variable) 
# v_out 的梯度是d(v_out)/d(variable) = 1/4*2*variable = variable/2
# max_out = max(variable*variable) 
# max_out 的梯度是d(max_out)/d(variable) = max(2*variable)
# mean,max之类的函数不参与梯度运算，原样保留
# max simply selects the greatest value and ignores the others, so max is the identity operation for that one element. 
# Therefore the gradient can flow backwards through it for just that one element.


print(variable.grad)    # 初始 Variable 的梯度
'''
 0.5000  1.0000
 1.5000  2.0000
'''


