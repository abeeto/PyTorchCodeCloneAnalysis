# 本小节介绍Tensor的各种操作。


import torch


'''
  算术操作（加法）
'''
# 算术操作
# 在PyTorch中，同一种操作可能有很多种形式，下面用加法作为例子。
x = torch.empty(5, 3)
x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x, '自定义数据类型')

# 加法形式一
y = torch.rand(5, 3)
print(x + y, '加法形式一')

# 加法形式二
print(torch.add(x, y), '加法形式二')

# 指定输出
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result, '指定输出')

# 加法形式三、inplace
# adds x to y
y.add_(x) # PyTorch操作inplace版本都有后缀_, 例如x.copy_(y), x.t_()
print(y, 'adds x to y')

#以上几种操作结果一样


'''
  索引
'''
# 我们还可以使用类似NumPy的索引操作来访问Tensor的一部分，需要注意的是：索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
y = x[0, :]
print(y, '原始y')
y += 1
print(y, '修改后的y')
print(x[0, :], '源tensor也被改了') # 源tensor也被改了

'''
  改变形状
'''
# 用view()来改变Tensor的形状
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来,一共5列，行自动计算
print(x.size(), y.size(), z.size(), 'x,y,z')


# 注意view()返回的新Tensor与源Tensor虽然可能有不同的size，
# 但是是共享data的，也即更改其中的一个，
# 另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
x += 1
print(x, '改变x的值')
print(y, 'y的值也发生了改变') # 也加了1


# 所以如果我们想返回一个真正新的副本（即不共享data内存）该怎么办呢？
# Pytorch还提供了一个reshape()可以改变形状，
# 但是此函数并不能保证返回的是其拷贝，所以不推荐使用。
# 推荐先用clone创造一个副本然后再使用view
# 使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。
x_cp = x.clone().view(15)
x -= 1
print(x, 'x-1')
print(x_cp, 'x_cp:拷贝值')


# 另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number
x = torch.randn(1)
print(x, 'x')
print(x.item(), 'x.item()')