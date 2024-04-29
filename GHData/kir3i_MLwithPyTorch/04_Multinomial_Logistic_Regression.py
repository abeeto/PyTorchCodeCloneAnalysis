import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# hypothesis로 softmax 함수를 이용한다.
z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)

# cost function으로 Cross Entropy를 이용한다.
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

y = torch.randint(5, (3, )).long()  # 임시 정답 생성
print(f'y_train: {y}')

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
# cost = F.nll_loss(F.log_softmax(z, dim=1), y)
# cost = F.cross_entropy(z, y)
print(cost)


print('='*100)


# Training with High-level Implementation with nn.Module

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)   # W의 차원을 나타낸다. x가 n*4, result가 n*3이므로 4*3이 되어야 함.
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    hypothesis = model.forward(x_train)

    cost = F.cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch: {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')
