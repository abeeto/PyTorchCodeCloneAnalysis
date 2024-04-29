'''
需要理解dim的概念:dim=0,表示按照行的方式来进行comparsion
'''

# Tensor的比较运算
import torch

a = torch.rand(2,3)
b = torch.rand(2,3)
print(a)
print(b)

# torch.eq(input,other,out=None) # 按成员进行等式操作,相同返回True
print(torch.eq(a,b))

# torch.euqal(tensor1,tensor2) # 如果tensor1和tensor2有相同的size和elements,则为True
print(torch.equal(a,b))

# torch.ge(input,other,out=None) # input >= other
print(torch.ge(a,b))

# torch.gt(input,other,out=None) # input > other
print(torch.gt(a,b))

# torch.le(input,other,out=None) # input <= other
print(torch.le(a,b))

# torch.lt(input,other,out=None) # input < other
print(torch.lt(a,b))

# torch.ne(input,other,out=None) # input != other
print(torch.ne(a,b))

# torch.sort(input,dim=None,descending=False,out=None) # 对目标input进行排序,默认为升序，返回值以及原来元素的索引值
c = torch.tensor([[1,4,5,2,6],
                  [9,7,5,2,6]])
print(c.shape)
print(torch.sort(c,dim=0,descending=True)) # dim=0,表示数据竖直上排序,即在行上排序
print(torch.sort(c,dim=1,descending=True))

# torch.topk(input,k,dim=None,largest=True,sorted=True,out=None) # 沿着指定维度返回最大k个数值及索引值
d = torch.tensor([[1,5,7,3,9],
                  [7,2,9,1,8]])
print(d.shape)
print(torch.topk(d, k=2, dim=0))

# torch.kthvalue(input,k,dim=None,out=None) # 沿着指定维度返回第k个最小值及其索引值
e = torch.rand(2,3)
print(e)
print(torch.kthvalue(e,k=1,dim=0)) # 按照行来比较，分别比价三对数值，找出最小的三个
print(torch.kthvalue(e, k=2, dim=1)) # 按照列

# Tensor判断是否为finite/inf/nan
# 1.torch.isfinite(tensor)/torch.isinf(tensor)/torch.isnan(tensor) # 依次为有界，无界，none
# 2.返回一个标记元素是否为finite/inf/nan 的mask张量
f = torch.rand(2,3)
print(f)
print(torch.isfinite(f))
print(torch.isfinite(f/0))
print(torch.isinf(f/0))
print(torch.isnan(f))













