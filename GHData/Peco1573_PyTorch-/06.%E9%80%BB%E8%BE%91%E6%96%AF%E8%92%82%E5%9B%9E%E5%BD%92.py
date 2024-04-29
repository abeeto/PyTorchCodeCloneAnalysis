import torch
import numpy as np

"""首先，选定数据集，这次仍然使用一个最简单的数据集"""
X_data = torch.FloatTensor([[1], [2], [3]])
y_data = torch.FloatTensor([[0], [0], [1]])
'''第二步，设计模型'''


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return torch.sigmoid(y_pred)


model = LogisticRegression()
'''第三步，设计criterion和optimizer'''
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

'''第四步，执行for循环'''
for epoch in range(100):
    y_pre = model(X_data)
    loss = criterion(y_pre, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()











