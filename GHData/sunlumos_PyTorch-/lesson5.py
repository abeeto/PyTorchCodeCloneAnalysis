import torch
#数据作为矩阵参与Tensor计算
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

#固定继承于Module  nn表示neural network神经网络
class LinearModel(torch.nn.Module):
    #构造函数初始化
    def __init__(self):
        #调用父类的init  此为固定语句，写就完了
        super(LinearModel, self).__init__()
        #Linear对象包括weight(w)以及bias(b)两个成员张量
        self.linear = torch.nn.Linear(1,1)

    #前馈函数forward，对父类函数中的overwrite
    def forward(self, x):
        #调用linear中的call()，以利用父类forward()计算wx+b
        y_pred = self.linear(x)
        return y_pred
    #反馈函数backward由module自动根据计算图生成
model = LinearModel()

#构造的criterion对象所接受的参数为（y',y） 用于接受损失值
criterion = torch.nn.MSELoss(size_average=False)
#model.parameters()用于检查模型中所能进行优化的张量
#learningrate(lr)表学习率，可以统一也可以不统一
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    #前馈计算y_pred
    y_pred = model(x_data)
    #前馈计算损失loss
    loss = criterion(y_pred,y_data)
    #打印调用loss时，会自动调用内部__str__()函数，避免产生计算图
    print(epoch,loss)
    #梯度清零
    optimizer.zero_grad()
    #梯度反向传播，计算图清除
    loss.backward()
    #根据传播的梯度以及学习率更新参数
    optimizer.step()

 #Output
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

#TestModel
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('y_pred = ',y_test.data)