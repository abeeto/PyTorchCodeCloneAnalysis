# -*- coding: utf-8 -*-
"""
 Copyright (c) 2019 Masahiko Hashimoto <hashimom@geeko.jp>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
学習データリスト
(入力) (論理演算種別 one-hot)
       AND OR XOR
0, 0,  1,  0, 0    => 0 と 0 の論理積
"""
train_list = [
    # 論理積 (AND)
    [0., 0., 1., 0., 0.],  # 0
    [0., 1., 1., 0., 0.],  # 0
    [1., 0., 1., 0., 0.],  # 0
    [1., 1., 1., 0., 0.],  # 1
    # 論理和 (OR)
    [0., 0., 0., 1., 0.],  # 0
    [0., 1., 0., 1., 0.],  # 1
    [1., 0., 0., 1., 0.],  # 1
    [1., 1., 0., 1., 0.],  # 1
    # 排他的論理和 (XOR)
    [0., 0., 0., 0., 1.],  # 0
    [0., 1., 0., 0., 1.],  # 1
    [1., 0., 0., 0., 1.],  # 1
    [1., 1., 0., 0., 1.]   # 0
]

"""
答えリスト
※上の学習データリストの答えをリスト化したもの
"""
answer_list = [
    [0.], [0.], [0.], [1.],  # 論理積の答え
    [0.], [1.], [1.], [1.],  # 論理和の答え
    [0.], [1.], [1.], [0.]   # 排他的論理和の答え
]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dense_1 = nn.Linear(5, 32)
        self.dense_2 = nn.Linear(32, 32)
        self.dense_3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x


# モデル定義
model = Model()

# 誤差関数設定 ※二乗誤差
criterion = nn.MSELoss()

# 最適化の設定
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 学習リストを変換
x_list = torch.Tensor(train_list)
y_list = torch.Tensor(answer_list)

# 学習開始 ※500エポック
model.train()
for i in range(500):
    # 学習データを入力して損失地を取得
    output = model(x_list)
    loss = criterion(output, y_list)

    # 勾配を初期化してBackProp
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("[%d] loss: %f" % (i, loss))

# 評価
print("----- Test -----")
model.eval()
for x_data in train_list:
    x_in = torch.Tensor(x_data)
    y_out = model(x_in)

    if x_data[2] == 1.:
        y_str = "and"
    elif x_data[3] == 1.:
        y_str = "or"
    else:
        y_str = "xor"
    print("%d %s %d = %f" % (x_data[0], y_str, x_data[1], y_out))

