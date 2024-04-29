# coding: utf-8

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Numpyの表示形式
np.set_printoptions(precision=3, suppress=True)

# 手書き文字認識用データセット
digits = load_digits(n_class=10)

# 正規化
digits.data = digits.data / np.max(digits.data)

# ラベルを配列化
labels = []
for target in digits.target:
    zero_list = np.zeros(10)
    zero_list[target] = 1
    labels.append(zero_list)

# データ
print("data={}".format(digits.data[0]))

# ターゲット
print("target={}".format(digits.target[0]))

# ラベル
print("label={}".format(labels[0]))

# 画像の表示
#plt.imshow(digits.data[0].reshape((8, 8)), cmap="gray")
#plt.show()

# 学習用データセット
train = torch.utils.data.TensorDataset(torch.from_numpy(np.array(digits.data)).float(), torch.from_numpy(np.array(labels)).float())
train_loader = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)


# フィードフォワード・ネットワーク
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

# ネットワークの初期化
network = Net()

# 損失関数（平均2乗誤差）
criterion = nn.MSELoss()

# オプティマイザ（ADAM）
optimizer = optim.Adam(network.parameters(), lr=0.01)

# 学習
for epoch in range(3):
    for i, (inputs, labels) in enumerate(train_loader):
        
        #print("inputs: {}".format(inputs))
        #print("labels: {}".format(labels))            

        # 入力に対する出力を取得
        outputs = network(inputs)

        #print("outputs: {}".format(outputs))
        
        # 損失の取得
        loss = criterion(outputs, labels)
        
        # 勾配の初期化
        optimizer.zero_grad()
        
        # 勾配の計算（損失関数を微分）
        loss.backward()        

        # パラメータの更新
        optimizer.step()
        
        if (i % 30) == 0:
            print("Epoch={} Step={} Loss={:.3f}".format(epoch, i, loss))

# 検証
index = 0
input = train[index][0]
output = network(input)
print("input={}".format(input.detach().numpy()))
print("label={}".format(train[index][1].detach().numpy()))
print("output={}".format(output.detach().numpy()))

