import torch

# 构建数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 类似于实例化一个tensor对象，1.0给它，注意一定要有[]
w = torch.Tensor([1.0])
# w需要计算梯度
w.requires_grad = True

# 构建预测函数，输入x 输出x * w
def forward(x):
    return x * w

# 构建损失函数，
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())
 
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # pytorch是动态图机制，所以在训练模型时候，每迭代一次都会构建一个新的计算图。而计算图其实就是代表程序中变量之间的关系
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        # 取l中的数值，用的是l.item，累加的时候要用sum += l.item ,而不是l，否则容易内存消耗殆尽
        
        # backward()函数，将之前的所有梯度计算出来，并且存放在w中，因为w设置了w.requires_grad = True
        # 计算的是loss对L的导数值，存放在w.grad中
        l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item())
        # 为了避免计算图，使用.data进行更新
        w.data = w.data - 0.01 * w.grad.data   # 权重更新时，注意grad也是一个tensor
 
    # 对导数进行清零，方便下一次使用  .zero_()清零语句
        w.grad.data.zero_() # after update, remember set the grad to zero
 
    #输出轮数，损失值详情来查看训练情况  
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
 
print("predict (after training)", 4, forward(4).item())