import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

## set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

## set data
sample = " if you want you"

# make dict
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}

# set hyperparameters
dic_size = len(char_dic)
hidden_size = len(char_dic)  # 다른 값 안됨. 왜 안되지? 아마도 output size와 맞아야 해서 그런듯
learning_rate = 0.1
input_size = len(char_set)

# data setting
sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]  # 마지막 글자 빼고
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]  # 첫 글자 빼고

# transform to Tensor
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

## make model
model = nn.RNN(
    input_size, hidden_size, batch_first=True
)  # batch_first guarantees the order of output

## set optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

nb_epochs = 100
for epoch in range(nb_epochs):
    hypothesis, _status = model(X)

    cost = F.cross_entropy(hypothesis.view(-1, input_size), Y.view(-1))
    # batch size를 제하고 cost를 구하기 위해 view 적용
    # print(hypothesis.shape)  # (1, 6, 5)
    # print(Y.shape)  # (1, 6)
    # print(hypothesis.view(-1, input_size).shape)  # (6, 5)
    # print(Y.view(-1).shape)  # (6)
    # print(cost)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    result = torch.argmax(hypothesis, dim=-1)
    result_str = "".join(
        [char_set[c] for c in torch.squeeze(result)]
    )  # batch size를 제하기위해 squeeze 적용
    print(f"epoch: {epoch+1:3d}/{nb_epochs} result: {sample[0]+result_str} ")


"""
학습을 몇 번 돌려보니 될 때 있고 안 될 때 있다.
바닐라라서 그런지 정확도가 떨어지는 것 같다.
epoch을 늘린다고 반드시 해결되지도 않았다.
"""
