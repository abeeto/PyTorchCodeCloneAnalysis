import torch
import numpy as np
from sklearn.datasets import load_breast_cancer
'''第一步，处理数据'''
X_data, y_data = load_breast_cancer(return_X_y=True)
y_data = y_data.reshape(-1, 1)
X_data = torch.from_numpy(X_data).float()
y_data = torch.from_numpy(y_data).float()
print(X_data.shape, y_data.shape)
'''第二步，构建模型'''


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(30, 15)
        self.linear2 = torch.nn.Linear(15, 6)
        self.linear3 = torch. nn.Linear(6, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


model = Model()
'''第三步，criterion 和 optimizer'''
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
'''第四步，train'''
for epoch in range(100):
    y_pre = model(X_data)
    loss = criterion(y_pre, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()













