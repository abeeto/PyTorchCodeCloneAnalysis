'''
여태까지는 1 Layer 구조로 학습하는 구조였다.
근데 1 Layer로 학습이 안되는 문제도 아주 많다.
대표적으로 XOR 문제가 있다.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## 단일 레이어로 Learning하는 경우

# 실행하는 device 인식 (gpu인지 cpu인지)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# linear = nn.Linear(2, 1, bias=True) # W의 크기 ->  XW = Y (X는 4*2, Y는 4*1이므로 W는 2*1)
# sigmoid = nn.Sigmoid()
# model = nn.Sequential(linear, sigmoid).to(device)
class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)   # W의 크기 ->  XW = Y (X는 4*2, Y는 4*1이므로 W는 2*1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = myModel()
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 10000
for epoch in range(nb_epochs+1):
    # calculate hypothesis
    hypothesis = model(X)
    # calculate cost
    cost = F.binary_cross_entropy(hypothesis, Y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch: {epoch:5d}/{nb_epochs} Cost: {cost.item():.6f}')
        # 출력해보면 cost에 변화가 없음을 확인할 수 있다. (개선되지 않는다.)
        if epoch == nb_epochs:
            isCorrect = (hypothesis>=0.5) == Y
            accuracy = torch.mean(isCorrect.float())    # 50%
            print(f'hypothesis:\n{hypothesis}')
            print(f'target:\n{Y}')
            print(f'Accuracy: {accuracy*100:.3f}%') # 50%



print('='*100)


## 다중 레이어로 Learning하는 경우
# Multi Layer Perceptron
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 이번엔 class 이용하지 않고 직접 모두 구현
w1 = torch.Tensor(2, 2).to(device)
b1 = torch.Tensor(2).to(device)
w2 = torch.Tensor(2, 1).to(device)
b2 = torch.Tensor(1).to(device)

# 초기화를 안해주니까 틀렸다..?
torch.nn.init.normal_(w1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(w2)
torch.nn.init.normal_(b2)

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

# sigmoid 도함수
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

learning_rate = 1
nb_epochs = 10000
for epoch in range(nb_epochs+1):
    ## forward
    l1 = torch.add(torch.matmul(X, w1), b1) # layer 1 통과
    a1 = sigmoid(l1)                        # activation function

    l2 = torch.add(torch.matmul(a1, w2), b2)# layer 2 통과
    a2 = sigmoid(l2)                        # activation function  <- this is 'hypothesis'
    hypothesis = a2

    cost  = -torch.mean(Y*torch.log(hypothesis) + (1 - Y)*torch.log(1-hypothesis))  # binary cross entropy로 계산

    ## backward (Backpropagation)
    # 전체적으로 편미분의 과정을 거침
    # Loss derivative
    d_a2 = (a2 - Y) / (a2 * (1.0 - a2) + 1e-7)  # 1e-7은 DividedByZeroException을 방지하기 위함.
    # layer 2
    d_l2 = d_a2 * sigmoid_prime(l2)
    d_b2 = d_l2
    d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_b2)
    # layer 1
    d_a1 = torch.matmul(d_b2, torch.transpose(w2, 0, 1))
    d_l1 = d_a1 * sigmoid_prime(l1)
    d_b1 = d_l1
    d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_b1)

    ## Weight update
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * torch.mean(d_b1, 0)
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * torch.mean(d_b2, 0)

    if epoch % 100 == 0:
        
        print(f'epoch: {epoch:5d}/{nb_epochs} Cost: {cost.item():.6f}')
        if epoch == nb_epochs:
            # l1 = torch.add(torch.matmul(X, w1), b1)
            # a1 = sigmoid(l1)
            # l2 = torch.add(torch.matmul(a1, w2), b2)
            # hypothesis = sigmoid(l2)
            isCorrect = (hypothesis >= torch.FloatTensor([0.5])) == Y
            accuracy = isCorrect.float().mean()
            print(f'hypothesis:\n{hypothesis}')
            print(f'target:\n{Y}')
            print(f'Accuracy: {accuracy*100:.3f}%')

print('='*100)

## 다중레이어로 하는데 이번엔 편하게 클래스로..

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cude':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# bulid a model
linear1 = nn.Linear(2, 2, bias=True)
linear2 = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

'''
# 더 많은 layer를 사용하면 loss를 더 줄일 수도 있다.
# 4MLP (4 Multi Layer Perceptron)
linear1 = nn.Linear(2, 10, bias=True)
linear2 = nn.Linear(10, 10, bias=True)
linear3 = nn.Linear(10, 10, bias=True)
linear4 = nn.Linear(10, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid, linear4, sigmoid).to(device)
'''

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 10000
for epoch in range(nb_epochs+1):
    # calculate hypothesis
    hypothesis = model(X)
    # calculate cost
    cost = F.binary_cross_entropy(hypothesis, Y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch: {epoch:5d}/{nb_epochs} Cost: {cost.item():.6f}')
        if epoch == nb_epochs:
            isCorrect = (hypothesis>=0.5) == Y
            accuracy = torch.mean(isCorrect.float())    # 정확도 확인
            print(f'hypothesis:\n{hypothesis}')
            print(f'target:\n{Y}')
            print(f'Accracy: {accuracy*100:.3f}%')