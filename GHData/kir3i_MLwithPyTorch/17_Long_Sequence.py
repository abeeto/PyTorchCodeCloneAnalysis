"""
좀더 긴 문자열에 대해 RNN을 이용하여 예측하는 모델
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np

## set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

## set hyperparameters
sequence_length = 10
hidden_size = 10
learning_rate = 0.001

## set data
sentence = "   if you want to build a ship, don't drum up poeple together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."

char_set = list(set(sentence))
char_dict = {c: i for i, c in enumerate(char_set)}

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + 1 + sequence_length]
    # print(f"{i}: IN {x_str} -> OUT {y_str}") # test
    x_data.append([char_dict[c] for c in x_str])  # str을 index로 바꿈
    y_data.append([char_dict[c] for c in y_str])  # str을 index로 바꿈

x_one_hot = [np.eye(len(char_dict))[x] for x in x_data]  # one-hot으로 변환

X = torch.FloatTensor(x_one_hot).to(device)
Y = torch.LongTensor(y_data).to(device)

## define model
class LongSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(char_dict), bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        return self.fc(x)


hidden_size = len(char_dict)
model = LongSequence(len(char_dict), hidden_size, 2).to(device)

## set optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## Learning
nb_epochs = 10000
result_str = sentence[0]
for epoch in range(nb_epochs):
    hypothesis = model(X)
    cost = F.cross_entropy(
        hypothesis.view(-1, len(char_dict)), Y.view(-1)
    )  # 차원에서 batch 크기를 제거

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # check cost and tmp result
    if epoch % 1000 == 0:
        result = torch.argmax(hypothesis, dim=-1)
        result_str = sentence[0]
        for i, res in enumerate(result):
            if i == 0:
                result_str += "".join(char_set[x] for x in res)
            else:
                result_str += char_set[res[-1]]
        print(
            f"epoch: {epoch+1:4d}/{nb_epochs} cost: {cost.item():.6f} result: {result_str}"
        )

if result_str == sentence:
    print("SUCCESS")
else:
    print("FAILED")

"""
예시로 나온 문자열도 실패하는 모습을 자주 보였다.
learning_rate와 hidden_size의 크기를 조정하며 해결해보려 시도했지만 잘 안 됐다.
문자열의 첫 글자만 틀리는 경향이 있었기 때문에 제일 앞에 패딩을 넣자 패딩 부분을 제외하곤 모두 맞추는 모습을 보였다.
한 칸으로는 부족했고 세 칸 정도 패딩을 넣어주니 잘 맞췄다.
"""
