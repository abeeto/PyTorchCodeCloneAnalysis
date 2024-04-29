import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# data setting
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 학습 data setting
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer setting
optimizer = optim.SGD([W, b], lr=1)

# Learning
nb_epochs = 1000
for epoch in range(nb_epochs):
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    # hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    cost = torch.mean(-(y_train * torch.log(hypothesis) + (1-y_train) * torch.log(1-hypothesis)))
    # cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1:4d}/{nb_epochs} Cost: {cost:.6f}')

# Evaluation
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
prediction = hypothesis >= torch.FloatTensor([0.5])
correct_prediction = prediction.float() == y_train
print(correct_prediction)

print('='*100)



# Higher implementation with class

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(accuracy)
        print(f'epoch: {epoch+1:4d}/{nb_epochs} Cost: {cost:.6f} Accuracy {accuracy*100:2.2f}%')