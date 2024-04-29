# 运算的内存开销


# 索引操作是不会开辟新内存的，
# 而像y = x + y这样的运算是会新开内存的，然后将y指向新内存。
# 为了演示这一点，我们可以使用Python自带的id函数：如果两个实例的ID一致，
# 那么它们所对应的内存地址相同；反之则不同。

import torch

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before, 'y = y + x') # False 


# 如果想指定结果到原来的y的内存，
# 我们可以使用前面介绍的索引来进行替换操作。
# 在下面的例子中，我们把x + y的结果通过[:]写进y对应的内存中。
# 通过切片的方式赋值属于浅拷贝
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before, 'y[:] = y + x') # True

# 我们还可以使用运算符全名函数中的out参数或者自加运算符+=(也即add_())达到上述效果，
# 例如torch.add(x, y, out=y)和y += x(y.add_(x))。（和切片的方式效果一样）
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before, 'torch.add(x, y, out=y)') # True
# 注：虽然view返回的Tensor与源Tensor是共享data的，
# 但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），
# 二者id（内存地址）并不一致。