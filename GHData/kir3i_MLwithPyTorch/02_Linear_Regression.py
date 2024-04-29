import torch

# Linear Regression

# This is data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True) # 학습시킬 변수
b = torch.zeros(1, requires_grad=True) # 학습시킬 변수

# This is optimizer (using SGD)
# SGD = stochastic gradient descent
optimizer = torch.optim.SGD([W, b], lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs):
    
    # calculate hypothesis
    hypothesis = x_train * W + b

    # cost function (loss function)
    cost = torch.mean((hypothesis - y_train) ** 2)

    print(f'epoch {epoch+1:4d}/{nb_epochs} W: {W.item():.3f} b: {b.item(): 3f} Cost: {cost.item():.6f}')
    # Gradient descent
    optimizer.zero_grad()   # Gradient 초기화
    cost.backward()         # Gradient 계산
    optimizer.step()        # cost값에 따라 학습 (결과 개선)


print('='*100)


# Multivariate Linaer Regression

'''
class MultivartiateLinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)
'''

x_train = torch.FloatTensor([[73, 80, 75],
                                [93, 88, 93],
                                [89, 91, 90],
                                [96, 98, 100],
                                [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# model = MultivariateLinearRegressionModel()

# optimizer 정의
optimizer = torch.optim.SGD([W, b], lr=(1e-5))
# optimizer = torch.optim.SGD(model.paameters(), lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs):
    # calculate hypothesis: matmul! (행렬 곱으로 계산)
    hypothesis = x_train.matmul(W) + b
    # hypothesis = model(x_train)

    # calculate cost function
    cost = torch.mean((hypothesis - y_train) ** 2)
    # cost = torch.nn.functional.mse_loss(hyp9othesis, y_train)

    # gradient descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #target: 152, 185, 180, 196, 142
    print(f'epoch {epoch+1:4d}/{nb_epochs} hypothesis: {hypothesis.squeeze().detach()} Cost: {cost:.6f}')



print('='*100)



# Minibatch Gradient Descent
# 많은 양의 data를 다룰 땐 한 번에 모든 data를 학습시키기 어렵다.
# 따라서 균일하게 나눠서 학습시키는 방법을 사용한다.

class MultivartiateLinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

# dataset 선언, 쪼개기
dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 선언
model = MultivartiateLinearRegressionModel()

# optimizer 선언
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        # calculate hypothesis
        hypothesis = model(x_train)

        # calculate cost function
        cost = torch.nn.functional.mse_loss(hypothesis, y_train)

        # Gradient descent
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f'epoch: {epoch+1:4d} Batch {batch_idx+1}/{len(dataloader)} Cost: {cost:.6f}')