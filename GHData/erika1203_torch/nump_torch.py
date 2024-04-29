'''
2018-09-15
学习pytorch第一课
torch可以代替numpy，用法类似
'''

import torch
from torch.autograd import Variable
import numpy as np

#torch的tensor数据类型
np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tensor2array=torch_data.numpy()
print('\nnumpy',np_data,'\ntorch',torch_data,'\ntensor2array',tensor2array)

#torch的数学运算
#http://pytorch.org/docs/torch.html#math-operations

data=[[1,2],[3,4]]
tensor=torch.FloatTensor(data) #32bit
print('\nnumpy:',np.matmul(data,data),  #矩阵乘法
    '\ntorch:',torch.mm(tensor,tensor))

#pytorch的变量单元
variable=Variable(tensor,requires_grad=True)
print(tensor,'\n',variable)

t_out=torch.mean(tensor*tensor)
v_out=torch.mean(variable*variable)
print(t_out,'\n',v_out)

#v_out=1/4*sum(variable*variable)
#d(v_out)/d(variable)=1/4*2variable=variable/2
v_out.backward()
print(variable.grad)

print(variable,'\n',variable.data,'\n',variable.data.numpy())




