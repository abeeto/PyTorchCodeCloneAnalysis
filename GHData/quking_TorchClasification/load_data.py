import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import seaborn as sns
import pandas as pd


n_data = torch.ones(100, 2)
x1 = torch.normal(n_data*2, 1)
y1 = torch.ones(100).unsqueeze(1)

x2 = torch.normal(-2*n_data, 1)
y2 = torch.zeros(100).unsqueeze(1)

x = torch.cat((x1, x2), 0)
y = torch.cat((y1, y2), 0).long()

x, y = Variable(x), Variable(y)

sns.set()
x_plot = x.data.numpy()[:,0]
y_plot = x.data.numpy()[:,1]
special = y.data.numpy()[:,0]

pdata = {"x_plot": x_plot, "y_plot": y_plot, "special": special}
df = pd.DataFrame(pdata)
sns.relplot(x="x_plot", y="y_plot", hue="special", data=df)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(2, 10)
        self.predict = nn.Linear(10, 2)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x


net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
for i in range(100):
    predict = net(x)
    loss = loss_func(predict, y.squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print("损失值：", loss)
        prediction = torch.max(F.softmax(predict), 1)[1]
