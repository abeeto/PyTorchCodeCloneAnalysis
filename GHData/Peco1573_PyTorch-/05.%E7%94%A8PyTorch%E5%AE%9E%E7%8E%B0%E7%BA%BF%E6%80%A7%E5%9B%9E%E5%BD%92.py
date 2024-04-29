import torch
import numpy as np
"""第一步，准备数据。使用最简单的数据"""
X_data = torch.FloatTensor([[1, 2, 3]]).T  # 新版本的Tenser有七种cpu Tenser和八种gpu Tenser。定义的时候就写仔细了。
y_data = torch.FloatTensor([[2, 4, 6]]).T  # 在64位电脑上，torch.Tenser其实是long类型的float。
print(X_data.shape)
print(y_data.shape)

'''第二步，构建模型。继承自torch.nn.Module'''


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pre = self.linear(x)  # 这里，直接在linear这个对象后面加括号，是因为torch.nn.Linear这个类在定义的时候就是callable, 内置__call__方法。
        return y_pre


model = LinearModel()       # 由于整个nn.Module都有内置的__call__函数，因此全部是callable的。model也是callable的。

'''第三步，构造损失函数和优化器'''
criterion = torch.nn.MSELoss(reduction='sum')   # mse损失函数就是只要有y,有y_pred ，就能计算损失。average就是是否除以样品数。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 传入的参数model.parameters()也是Module中定义的方法。在这里就是w和b.
'''optim.SGD中需要传递的是   1，params权重参数 2，lr学习率  3，momentum动量， 4，dampening， 5，weight_decay 6，nesterov=False'''

'''第四步，训练循环，training cycle'''
for epoch in range(100):
    y_pred = model(X_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)  # 这里注意，loss不是一个数值，它也是个对象。只是print的时候会调用__str__方法，不会产生计算图.或者可以输出loss.item()也不错。

    optimizer.zero_grad()   # 每次backward（）所得的梯度会积累。如果不想使其积累，就在每次backward前加这句 optimizer.zero_grad()
    loss.backward()
    optimizer.step()      # 这句话是参数更新。根据所选criterion自动更新权重参数

print('w=', model.linear.weight.item())
print('b', model.linear.bias.item())
print("Test/t")
X_test = torch.FloatTensor([[4]])
y_test = model(X_test)
print(y_test)


