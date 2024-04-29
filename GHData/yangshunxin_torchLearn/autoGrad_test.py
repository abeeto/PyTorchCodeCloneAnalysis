import torch

def autoGradeTest():
    # 自动求导
    # PyTorch中，所有神经网络的核心是 autograd包。先简单介绍一下这个包，然后训练我们的第一个的神经网络
    # autograd包为张量上的所有操作提供了自动求导机制。
    # 它是一个在运行时定义(define - by - run）的框架，
    # 这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的.
    # 张量
    # torch.tensor 是这个包的核心类，如果设置它的属性 .requires_grad为True, 那么它将会追踪对于该张量的所有操作。
    # 当完成计算后可以 通过调用 .backward(), 来自动计算所有的梯度。这个张量的所有梯度会自动累加到.grad属性。
    #　要阻止一个张量被跟踪历史，可以调用.detach() 方法将其计算历史分离，并阻止它未来的计算记录被跟踪。
    # 为了防止跟踪历史记录（和使用内存）， 可以将代码块包装在 with torch.no_grad(): 中，
    # 在评估模型时，特别有用， 因为模型可能具有 requires_grad = True 的可训练参数，但是我们不需要在此过程中对他们进行梯度计算。

    # 还有一个类对于 autograd的实现非常重要：Function
    # Tensor 和 Function 相互连接生成了一个无圈图（acyclic graph）, 它编码了完整的计算历史。每个张量都有一个.grad_fn属性，
    # 该属性引用了创建Tensor自身的Function（除非这个张量是用户手动创建的，即这个张量的grad_fn是None）

    # 如果需要计算导数，可以在Tensor上调用.backward(). 如果Tensor是一个标量（即它包含一个元素的数据），则不需要为backward()指定任何参数，
    # 但是如果它有更多的元素，则需要指定一个gradient参数，该参数是形状匹配的张量。

    # 创建一个张量并设置requires_grad=True用来追踪其计算历史
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    # 对这个张量做一次运算
    y = x + 2
    print(y) # y 是通过 加法操作来创建的
    # y是计算的结果，所以它有grad_fn属性
    print(y.grad_fn)
    # 对y进行更多操作
    z = y * y * 3
    out = z.mean()
    print(z, out)


    # .requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是 False
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b, b.grad_fn)

def gradCalcTest():
    # 现在开始进行反向传播，因为out是一个标量，因此out.backward()和out.backward(torch.tensor(1.))等价。
    # out.backward()  输出导数 d(out)/dx  print(x.grad)
    # 创建一个张量并设置requires_grad=True用来追踪其计算历史
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    # 对这个张量做一次运算
    y = x + 2
    print(y) # y 是通过 加法操作来创建的
    # y是计算的结果，所以它有grad_fn属性
    print(y.grad_fn)
    # 对y进行更多操作
    z = y * y * 3
    out = z.mean()
    print(z, out)
    out.backward()
    # 我们的得到的是一个数取值全部为4.5的矩阵, 计算out张量o
    # d(out)/d(x1) = 4.5
    # 数学上，x,y都是向量，若有向量值函数y=f(x), y对x的梯度是一个雅可比矩阵
    #　通常来说，torch.autograd是计算雅可比向量积的一个"引擎"。 也就是说，给定任意向量 v, 计算乘积v(T)*J,
    # 此处导出 链式求导法则，能够很快的计算出，后续输入变量对 out的倒数

    print(x.grad)
    # 雅可比向量积的这一特性使得将外部梯度输入到具有非标量输出的模型中变得非常方便
    # 现在我们来看一个雅可比向量积的例子
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)
    # 在这种情况下，y不再是标量。torch.autograd不能直接计算完整的雅可比矩阵，但是如果我们只想要雅可比向量积，只需将这个向量作为参数传给backward
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)

    print(x.grad)
    # 也可以通过将代码块包装在 with torch.no_grad(): 中，来阻止autograd跟踪设置了 .requires_grad=True 的张量的历史记录。
    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)
    # 在底层，每一个原始的自动求导运算实际上是两个在Tensor上运行的函数。
    # 其中，forward函数计算从输入Tensors获得的输出Tensors。而backward函数接收输出Tensors对于某个标量值的梯度，并且计算输入Tensors相对于该相同标量值的梯度

if __name__=="__main__":
    # autoGradeTest()
    gradCalcTest()
