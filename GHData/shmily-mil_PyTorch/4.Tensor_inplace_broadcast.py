'''

in-place:就地操作,就是不允许使用临时变量，就是累加的意思,x = x + y,add_,sub_,mul_等

广播机制:张量参数可以自动扩展为相同大小
满足条件:1.每个张量至少有一个维度
        2.满足右对齐,右边第一个相等或者一个是1,则满足右对齐
        3.torch.rand(2,1,1)+torch.rand(3)

'''

import torch

a = torch.rand(2,3)
b = torch.rand(3)
c = a + b
print(a)
print(b)
print(c)

