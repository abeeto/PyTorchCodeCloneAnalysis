"""
time series data: 시계열 데이터
시계열 데이터를 다루는 RNN 모델을 만든다.
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

## set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

## set hyperparameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
nb_epochs = 500

## set data
xy = np.loadtxt("data-02-stock_daily.csv", delimiter=",")
xy = xy[::-1]  # reverse order

train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]  # 70%는 학습set
test_set = xy[train_size - seq_length :]  # 30%는 테스트set

# scale에 대한 문제 해결
def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denomiator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denomiator + 1e-7)


# data 가공
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i : i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]
        # print(f"{_x} -> {_y}")
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)


train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)

trainX, trainY = map(torch.FloatTensor, build_dataset(train_set, seq_length))
testX, testY = map(torch.FloatTensor, build_dataset(test_set, seq_length))

## define model
class TimeseriesRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        return self.fc(x[:, -1])  # 제일 마지막만 fc로 돌림 (?)


model = TimeseriesRNN(data_dim, hidden_dim, output_dim, 1)

## set optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## Learning
for epoch in range(nb_epochs):
    hypothesis = model(trainX)
    cost = F.mse_loss(hypothesis, trainY)  # MSE Loss 이용

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print(f"epoch: {epoch+1:3d}/{nb_epochs} cost: {cost.item():.6f}")

## test
plt.plot(testY)
plt.plot(model(testX).data.numpy())
plt.legend(["original", "hypothesis"])
plt.show()

"""
얼추 비슷한 모양의 그래프가 나왔다.
그래프를 정답에 가깝게 fitting하는 모델이라 그런지
loss 계산 시 MSE Loss를 사용했다.
"""
