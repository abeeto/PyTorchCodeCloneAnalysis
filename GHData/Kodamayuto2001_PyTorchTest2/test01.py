#参考文献
#https://www.kikagaku.ai/tutorial/basic_of_computer_vision/learn/pytorch_convolution

"""
import torch,torchvisionの環境構築方法
このPCにはグラフィックボードが入っていないので,CPUのPyTorchを使う
conda install pytorch-cpu torchvision-cpu -c pytorch
"""
import torch
import torchvision#MNISTのデータセット

from torchvision import transforms#Pytorchようにデータを変形するライブラリ
#print(torch.__version__)
#print(torchvision.__version__)

import numpy as np 

"""
import matplotlibの環境構築方法
conda install -c anaconda matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib

"""
import cv2 の環境構築方法
conda install -c conda-forge opencv
"""
#画像出力で使う
#import cv2

"""
多分使うかなと思った
"""
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
#torch.nn == ネットワークを作るために必要
#torch.nn.functional == 畳み込み関数

#データ読み込み時の処理
#PyTorchでは、データをtorch.Tensor型にして扱う
#PyTorch公式
#https://pytorch.org/docs/stable/torchvision/transforms.html
transform = transforms.Compose([transforms.ToTensor()])


#torchvision.datasetsにはデフォルトで用意されているデータセットがいくつもある
#PyTorch公式
#https://pytorch.org/docs/stable/torchvision/datasets.html
#学習用データセットの読み込み
train = torchvision.datasets.MNIST(root="Resources/",train=True,download=True,transform=transform)

#データセットの中身を確認
print("データセットの中身\n{}".format(train))

#サンプル数
print("サンプル数:{}".format(len(train)))

#入力値と目標値をタプルで格納
print("データセットのタイプ：{}".format(type(train[0])))

#入力値
print("データセット中身(数字)：{}".format(train[0][0]))

#目標値＝＝ラベル
print("データセットのラベル:{}".format(train[0][1]))

#入力値のサイズ
print("入力値のサイズ{}".format(train[0][0].shape))

#PyTorchの特性確認
#PyTorchでは(channels,height,width)の順に格納　※"(height,width,channels)"ではない
#確認
c,h,w = train[0][0].shape

print("channels={}".format(c))
print("height  ={}".format(h))
print("width   ={}".format(w))

#データセットの中身を数字ではなく画像で確認したい
#Matplotlibを使う
#train[0][0]のデータ格納順番を変更
# (0:channels, 1:height, 2:width) -> (1:height, 2:width、0:channels)
img = np.transpose(train[0][0],(1,2,0))
#グレースケールつまり白黒画像で表示
#チャネルサイズはなくす
img = img.reshape(img.shape[0],img.shape[1])
print("グレースケール用チャネル排除結果：{}".format(img.shape))
print("チャネル：{}".format(c))


plt.imshow(img,cmap="gray")
#表示
plt.show()

#---------------一連の流れの確認---------------
#特徴量抽出
#--畳み込み
#--プーリング
#--全結合層
x = train[0][0]
print(x.shape)

print("-------------------------------------------------------------------------")

#畳み込み層の定義
#用語解説
#https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c
#https://torch.classcat.com/2017/04/14/pytorch-tutorial-neural-networks/
#https://deepage.net/deep_learning/2016/11/07/convolutional_neural_network.html
#PyTorch公式
#https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html
conv = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
"""
in_channels (int)               –入力画像のチャネル数
out_channels (int)              –たたみ込みによって生成されたチャネルの数
kernel_size (int or tuple)      – たたみ込みカーネルのサイズ
stride (int or tuple, optional) –畳み込みのストライド。デフォルト：1
padding (int or tuple, optional)–入力の両側に追加されるゼロパディング。デフォルト：0
padding_mode (string, optional) - 'zeros' または。デフォルト：'reflect''replicate''circular''zeros'
dilation(int or tuple, optional)–カーネル要素間の間隔。デフォルト：1
groups (int, optional)          –入力チャネルから出力チャネルへのブロックされた接続の数。デフォルト：1
bias (bool, optional)           –の場合True、学習可能なバイアスを出力に追加します。デフォルト：True
"""
#畳み込み層を宣言した時点でフィルタの重みがランダムに割り振られている
print("conv.weight=\n{}".format(conv.weight))
print("conv.weight.shape=\n{}".format(conv.weight.shape))
print("conv.bias=\n{}".format(conv.bias))
print("conv.bias.shape=\n{}".format(conv.bias.shape))

#(batchsize,channels,height,width)
#入力層
x = x.reshape(1,1,28,28)
#👇
#畳み込み層
x = conv(x)
print("畳みこまれたもの：\n{}".format(x))
print(x.shape)
#👇
#プーリング処理
x = F.max_pool2d(x,kernel_size=2,stride=2)
print(x.shape)
#👇
#畳み込み層
#👇
#プーリング層
#👇
#畳み込み層
#👇
#プーリング層
#👇
#畳み込み層
#👇
#全結合層

#全結合層と結合
print("-------------------------------------------------------------------------")
print("channels:{}".format(x.shape[1]))
print("heights :{}".format(x.shape[2]))
print("width   :{}".format(x.shape[3]))
#Flatten 
#4階のテンソルを1階のテンソル(ベクトル)に変換
x_shape = x.shape[1] * x.shape[2] * x.shape[3]
print("x_shape :{}".format(x_shape))

# 今回はベクトルの要素数が決まっているため、サンプル数は自動で設定
# -1 とするともう片方の要素に合わせて自動的に設定される
#サイズ変更をする際には torch.view() 関数を使う
x = x.view(-1, x_shape)
print("x.shape :{}".format(x.shape))

#全結合層の定義 ノード数を 10
fc = nn.Linear(x_shape,10)#x_shape=784

#出力層
#線形変換
x = fc(x)
print(x)
print(x.shape)







