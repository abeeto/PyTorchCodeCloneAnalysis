'''
https://www.cnblogs.com/wj-1314/p/9830950.html
'''

#build a simple neural network
import torch

#导入包
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
'''
batch_n是在一个批次中输入数据的数量
每个数据包含的数据特征有input_data个
hidden_layer用于定义经过隐藏层后保留的数据特征的个数
    因为我们的模型只考虑一层隐藏层，所以在代码中仅仅定义了一个隐藏层的参数
output_data是输出的数据,可以将输出的数据看作一个分类结果值得数量

一个批次的数据从输入到输出的完整过程是：
先输入100个具有1000个特征的数据，经过隐藏层后变成100个具有100个特征的数据，
再经过输出层后输出100个具有10个分类结果值的数据，在得到输出结果之后计算损失并进行后向传播，
这样一次模型的训练就完成了，然后训练这个流程就可以完成指定次数的训练，并达到优化模型参数的目的。
'''

#初始化权重
x = torch.randn(batch_n,input_data)
y = torch.randn(batch_n,output_data)

w1 = torch.randn(input_data,hidden_layer)
w2 = torch.randn(hidden_layer,output_data)
'''
在以上的代码中定义的从输入层到隐藏层，从隐藏层到输出层对应的权重参数，
同在之前说到的过程中使用的参数维度是一致的,
可以看到，在代码中定义的输入层维度为（100,1000），输出层维度为（100,10），
同时，从输入层到隐藏层的权重参数维度为（1000,100），从隐藏层到输出层的权重参数维度为（100,10）
'''

#定义训练次数和学习效率
epoch_n = 20
learning_rate = 1e-6
'''
由于接下来会使用梯度下降的方法来优化神经网络的参数，所以必须定义后向传播的次数和梯度下降使用的学习效率。
在以上代码中使用了epoch_n定义训练的次数
在优化的过程中使用的学习效率为learning_rate
'''

#梯度下降优化神经网络的参数
for epoch in range(epoch_n):
    h1 = x.mm(w1)  # 100*1000
    h1 = h1.clamp(min=0)
    y_pred = h1.mm(w2)  # 100*10
    # print(y_pred)
 
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{} , Loss:{:.4f}".format(epoch, loss))
 
    gray_y_pred = 2 * (y_pred - y)
    gray_w2 = h1.t().mm(gray_y_pred)
 
    grad_h = gray_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1 = x.t().mm(grad_h)
 
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * gray_w2
'''
循环内的是神经网络模型具体的前向传播和后向传播代码。参数的优化和更新使用梯度
'''
'''
以上代码通过最外层的一个大循环来保证我们的模型可以进行20层训练，
循环内的是神经网络模型具体的前向传播和后向传播代码，参数的优化和更新使用梯度下降来完成。
在这个神经网络的前向传播中，通过两个连续的矩阵乘法计算出预测结果，
在计算的过程中还对矩阵乘积的结果使用clamp方法进行裁剪，
将小于零的值全部重新赋值于0，这就像加上了一个ReLU激活函数的功能。

前向传播得到的预测结果通过 y_pred来表示，在得到了预测值后就可以使用预测值和真实值来计算误差值了。
我们用loss来表示误差值，对误差值的计算使用了均方误差函数。
之后的代码部分就是通过实现后向传播来对权重参数进行优化了，
为了计算方便，我们的代码实现使用的是每个节点的链式求导结果，
在通过计算之后，就能够得到每个权重参数对应的梯度分别是grad_w1和grad_w2。
在得到参数的梯度值之后，按照之前定义好的学习速率对w1和w2的权重参数进行更新，
在代码中每次训练时，我们都会对loss的值进行打印输出，以方便看到整个优化过程的效果，
所以最后会有20个loss值被打印显示。
'''