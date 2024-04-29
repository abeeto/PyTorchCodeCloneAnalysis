# -*- coding: utf-8 -*-
# weibifan 2022-10-2
#  自动微分，arch.autograde 自动梯度---所有带自动的都需要考虑，
# ①什么时间开始，触发条件是什么。 ②什么时间终止，终止条件是什么。

'''  Torch的核心，难点   2022-10-13

# Tensor对象，在创建时声明是可微的，或者后期通过设置函数使其可微。

# 只计算计算图的叶子节点。 输入是叶子节点，输出是root

梯度的计算分为2步：
第1步：当前Tensor变量的微分，比如loss=l(g), 此时只计算l的微分l`，不展开g
对于Tensor变量g的微分，g=g(p), 此时只计算g的微分g`，不展开p
这一步是自动计算，每次当前变量发生变化，都会计算。

第2步：损失函数，标量，调用backward()，链式计算，也就是l` * g` ，
对谁微分（比如w），链式计算结果放到 w的梯度里。
这一步是显式调用，调用才算。

torch.no_grad() 会暂停一次 第1步的微分及梯度计算。

基础：y=f(w)=wx+b     f'(.)=x
求梯度分为2步：①对w，求y或者f()的导数，得到f'(.) ②将x的值代入，得到梯度值f'(x)
在PyTorch中，针对每个表达式f(x)总能得到其微分表达式f'(.)，f'(.)存在grad_fn属性中，
变量的梯度值f'(x)自动计算，只要x发生变化就计算，放到Tensor变量y的grad里面。

问题1：任意函数的微分怎么求？
根据表达式形成符号表达式，然后获得表达式的微分。

问题2：链式求解中，中间数据存在哪里？

'''


import torch

# 梯度=曲线在某个位置的斜率。
# x和y是叶子节点，没有曲线，也就是没有斜率，没有梯度。也就是说x和y是常数。
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

# w和b是叶子节点，没有曲线，但是指定了梯度，说明w和b是参数。
w = torch.randn(5, 3, requires_grad=True) #参数矩阵 5*3
b = torch.randn(3, requires_grad=True)

# z和loss 有表达式，有曲线，有斜率，有梯度。
# 说明 需要分别对w和b求z和loss的偏导数
z = torch.matmul(x, w)+b  #注意顺序
print(z.requires_grad)  #默认为True

'''   weibifan 2022-10-12

z.sum().backward()
# w点的梯度，b点的梯度
print('w0=', w.grad)
print('b0=', b.grad)

RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

'''


#  感知器？
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 有曲线（表达式），就会有导数函数
print(f"Gradient function for w = {w.grad_fn}")  #结果为none，因为w没有更新
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# 对所有叶子节点（w和b）求loss的梯度
loss.backward()  #该函数会依次调用各个节点的微分函数。

# w点的梯度，b点的梯度
print('w=', w.grad)
print('b=', b.grad)

print(f"2: Gradient function for w = {w.grad_fn}")
# 关闭梯度计算？？？ ①测试阶段。②finetuning
'''
 一个上下文管理器，disable梯度计算。
 disable梯度计算对于推理是有用的，当你确认不会调用Tensor.backward()的时候。
 这可以减少计算所用内存消耗。这个模式下，每个计算结果的requires_grad=False，
 尽管输入的requires_grad=True。

等价于：
torch.no_grad()  #关闭梯度计算
try：
    z = torch.matmul(x, w)+b

torch.enable_grad()  #恢复梯度计算？？？
print(z.requires_grad)


'''

with torch.no_grad():
    z = torch.matmul(x, w)+b # z的梯度计算被关闭了。更新z时，不进行梯度计算
    # w和b没有更新，w和b的梯度计算未变
print('grad2 = ', z.requires_grad)

z = torch.matmul(x, w)+b # z更新了，z的梯度计算开启了。
z_det = z.detach() #
print('grad3 = ', z_det.requires_grad)