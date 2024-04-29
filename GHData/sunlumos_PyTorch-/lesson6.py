import torch.nn.functional as F
import torch

# todo 训练数据的准备
# 数据的准备中，从原来的线性数据，修改成分类
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0.0],[0.0],[1.0]])

# todo 设计相应的神经网络训练模型
#改用LogisticRegressionModel 同样继承于Module
class LogisticRegressionModel(torch.nn.Module):
    # 前面的代码段基本一致，没有改变
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        #对原先的linear结果进行sigmod激活
        # 调用function包里的sigmoid函数
        # 首先使用self.linear(x)来求函数，然后使用sigmoid对其进行一次处理，作为最后的输出
        # 和线性变换的区别就是多了是用sigmoid函数来进行处理
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

# todo 选择损失与优化器
#构造的criterion对象所接受的参数为（y',y） 改用BCE
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# todo 进行训练循环
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('y_pred = ',y_test.data)