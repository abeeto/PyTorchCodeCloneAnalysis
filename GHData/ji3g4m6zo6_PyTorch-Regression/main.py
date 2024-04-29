# coding=UTF-8

import torch
from torch.autograd import Variable
import torch.nn.functional as torch_functional
import matplotlib.pyplot as plt


# 定義神經網路
class Net(torch.nn.Module):
    # 初始化 參數為(輸入個數, 隱藏層神經元節點個數, 輸出個數)
    def __init__(self, n_feature, n_hidden, n_output):
        # 透過 Super Class 預先初始化自定義的 Net
        super(Net, self).__init__()

        # 隱藏層
        self.hidden = torch.nn.Linear(n_feature, n_hidden)

        # 輸出層
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 向前傳遞的過程
    def forward(self, x):
        # 先將 x 透過隱藏層加工輸出神經元個數，再用 activation function（激勵函數）優化
        hidden_output = torch_functional.relu(self.hidden(x))

        # 再將隱藏層的輸出放入輸出層，此處不使用 activation function ，因為會截斷輸出結果。
        predict_output = self.predict(hidden_output)
        return predict_output


if __name__ == "__main__":
    # 製作實驗用數據
    # x 製作成由 -1 到 1 的共一百個點的二維數據
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

    # y 為 x 的二次方 上一些隨機點
    y = x.pow(2) + 0.2 * torch.rand(x.size())

    # 轉換成 Variable（神經網路只能接收 Variable ）
    x, y = Variable(x), Variable(y)

    # 數據可視化（設定 Matplotlib 持續印出數據）
    plt.ion()
    plt.show()

    # 初始化定義好的神經網路
    # 輸入一個(x)，十層隱藏層，預測輸出一個(y)
    net = Net(1, 10, 1)

    # SGD優化器，傳遞參數與學習效率，學習效率越高速度越快，但學習效果會因為梯度過大而權重(weights)離實際的最佳值越來越遠，數值通常為小於一
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    # 透過 MSELoss 計算誤差，透過 MSELoss 處理回歸問題即可
    loss_func = torch.nn.MSELoss()

    # 訓練三百次
    for t in range(300):
        prediction = net(x)

        # 計算預測值與真實值的誤差，預設值要在前，真實值在後
        loss = loss_func(prediction, y)

        # 優化
        # 1. 將所有參數的梯度降為零
        optimizer.zero_grad()

        # 2. 反向傳遞，將每個節點計算出梯度
        loss.backward()

        # 2. 以學習效率 0.5 優化梯度
        optimizer.step()

        if t % 5 == 0:
            plt.cla()

            # 印出原始數據
            plt.scatter(x.data.numpy(), y.data.numpy())

            # 印出預測數據
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

            # 印出誤差數據
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()
