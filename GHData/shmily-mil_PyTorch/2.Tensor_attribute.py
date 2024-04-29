'''

Tensor的属性:
1.每一个Tensor有torch.dtype,torch.device,torch.layout三种属性
2.torch.device标识了torch.Tensor对象在创建之后所存储在的设备名称,cpu或者gpu
3.torch.layout表示torch.Tensor内存布局的对象

# 定义稀疏的张量，torch.sparse_coo_tensor,coo类型表示了非零元素的坐标形式
# 稀疏的好处:模型简单,减少内存的开销
# 稀疏表示当前元素非零元素的个数

'''

import torch

 # a = torch.tensor([1,2,3],
 #                  dtype=torch.float32,
 #                  device=torch.device("cpu"))

# 稀疏张量的定义
# i = torch.tensor([[0,1,1],[2,0,2]]) # 非零元素坐标值，坐标定义两个集合，一个表示x的集合，一个表示y的集合，所有这个坐标为(0,2),(1,0),(1,2),对于应的三个元素为3,4,5
# v = torch.tensor([3,4,5],dtype=torch.float32) # 具体值
# 格式:x = torch.sparse_coo_tensor(非零元素坐标值,具体值,shape)
# x = torch.sparse_coo_tensor(i,v,[2,4])

# 在不指定张量为稀疏的时候创建的张量都是稠密的张量
i = torch.tensor([[0,1,2],[0,1,2]]) # 非零数据的坐标
v = torch.tensor([1,2,3]) # 数据
b = torch.sparse_coo_tensor(i,v,(4,4))
c = b.to_dense() # 稀疏转化为稠密
print(b)
print(f'稀疏张量b转化为稠密张量c{c}')