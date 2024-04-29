# coding: utf-8
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os

OUTPUT_CHANNELS = 64
# OUTPUT_CHANNELS = 100
HIDDEN_LAYER_NUM = 10

class AgentNet(nn.Module):
    """
    PyTorchではinitに使用する層の種を記述するのが標準的
    """
    def __init__(self):
        super(AgentNet, self).__init__()

        ## 中間層の活性化関数をReLUにする
        self.relu = nn.ReLU()

        ## 三種類の畳み込み層を用いる
        # type1: 入力のチャンネル数:１・出力のチャンネル数:OUTPUT_CHANNELS(出力のチャンネル数)
        # type2: 入力のチャンネル数:・出力のチャンネル数:OUTPUT_CHANNELS(出力のチャンネル数)
        # type3: 入力のチャンネル数:OUTPUT_CHANNELS・出力のチャンネル数:1(入力のチャンネル数, 譜面データに合わせた)
        # nn.Conv2d(入力のチャンネル数, 出力のチャンネル数, 正方形フィルタのサイズ, strides, padding, バイアスの有無)
        self.conv_type1 = nn.Conv2d(1, OUTPUT_CHANNELS, 3, 1, 1, bias=False) # 初期層
        self.conv_type2 = nn.Conv2d(OUTPUT_CHANNELS, OUTPUT_CHANNELS, 3, 1, 1, bias=False)  # 第2~9層
        self.conv_type3 =  nn.Conv2d(OUTPUT_CHANNELS, 1, 1, 1, 0, bias=True)  # 最終層

        ## バッチ正規化
        self.bn = nn.BatchNorm2d(OUTPUT_CHANNELS, affine=False)

    """
    ネットワークの最終出力をする。
    出力はsoftmax関数にて行われ、8x8譜面->(1, 64)の確率値となる
    (確率値が最も高い場所に石を配置すること)。
    """
    def forward(self, x):
        ## 第１層
        x = self.conv_type1(x)
        x = self.bn(x)
        x = self.relu(x)

        ## 第２～９層 TODO:これをまとめてHIDDEN_LAYER_NUMの数だけ層を作れるように
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_type2(x)
        x = self.bn(x)
        x = self.relu(x)

        ## 第１０層
        x = self.conv_type3(x)

        size = x.data.shape[0]  # size = 100(MINIBATCH SIZEと同じ)
        x = torch.reshape(x, (size, 64))  # (100, 64)の形に

        ## softmax出力は使っちゃダメ(pytorchでは損失関数のところに自動的に組み込まれる)
        y = F.softmax(x, dim=1)

        return x
