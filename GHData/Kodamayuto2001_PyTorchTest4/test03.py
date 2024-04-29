"""
ニューラルネットワーク構築　torch.nn パッケージを用いる
モデルの構築は,nn.Moduleを継承して定義する
nn.Moduleにはレイヤーと出力を返すforwardメソッドを定義する
nnはモデルを定義したり、モデルの微分計算をするので、autogradに依存するパッケージです。
nn.Moduleには、層とメソッドforward(input)が含まれており、forward(input)はoutputを返します。
FNN フィールドフォワード・ニューラルネットワーク 順伝播型ニューラルネットワーク
RNN リカレント・ニューラルネットワーク 再帰型ニューラルネットワーク
今回はシンプルなフィールドフォワードニューラルネットワークの構築
・いくつかの学習可能なパラメータ(または重み)を持つニューラルネットワークを定義する
・入力のデータセットを反復する
・ネットワークを介して入力を処理する
・損失を計算する(出力が正しい値からどれだけ離れているか)
・グラデーションをネットワークのパラメータに反映
・通常は単純な更新ルールを使用して、ネットワークの重みを更新します。weight = weight - learning_rate + gradient
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Linear(784,400)
        self.layer2 = nn.Linear(400,2)


    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


net = NeuralNetwork()
print(net)

#net.parameters()をリストに型変換することでパラメータが取り出せる
params = list(net.parameters())
#パラメータの種類の数
#今回はlayer1の重み、layer1のバイアス、layer2の重み、layer2のバイアスの4つ
print("パラメータの種類の数:{}".format(len(params)))
print(params[0].size())
print(params[1].size())
print(params[2].size())
print(params[3].size())








