import torch
import numpy as np
#读取文件，一般GPU只支持32位浮点数，使用,来进行分割
# 神经网络中常使用float32
xy = np.loadtxt('digits.csv.gz', delimiter=',', dtype = np.float32)
#所有行，第一列开始，-1表示最后一列不要
x_data = torch.from_numpy(xy[:, :-1])
#单取-1列作为矩阵
y_data = torch.from_numpy(xy[:-1, [-1]])
#取-1行的测试集部分
test_data = torch.from_numpy(xy[[-1], :-1])
pred_test = torch.from_numpy(xy[[-1],[-1]])
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 输入为8维，输出是6维
        self.linear1 = torch.nn.Linear(8, 6)
        # 输入为6维，输出是4维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # sigmoid函数  与上次不同的是，这次使用的是nn.sigmoid一个模块
        self.sigmoid = torch.nn.Sigmoid()
        # 后面需要修改成其他的激活函数就修改上面这行即可
        # self.activate = torch.nn.ReLu()

    def forward(self, x):
        # 如果模型是序列式的，从上往下走，那就用x就可以，以免每一层都设置一个变量，容易出错
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    #Forward 并非mini-batch的设计，只是mini-batch的风格
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Update
    optimizer.step()

print("test_pred = ", model(test_data).item())
print("infact_pred = ", pred_test.item())