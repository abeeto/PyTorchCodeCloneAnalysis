# Data Preprocessing
# Learning 전에 Data를 정규분포 N(0, 1)로 표준화해서 계산.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def train(model, optimizer, x_train, y_train, nb_epochs=20):
    
    for epoch in range(nb_epochs + 1):
        # calculate hypothesis
        hypothesis = model(x_train)
        # calculate cost function
        cost = F.mse_loss(hypothesis, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f'epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)
# 정규화를 이용한다면 scale이 변하는 것이기때문에 learning rate를 조정해야 한다.
# 정규화를 하지 않고 그냥 값으로 하면 lr=1e-5로 해야한다. (안 그러면 overfitting 현상 발생)
train(model, optimizer, norm_x_train, y_train)
