# coding: utf-8
import load # load.pyの処理をimport
import network  # network.pyで定義されているクラスをimport
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# from torchsummary import summary  # ネットワーク構成の表示に使用(kerasのmodel.summary()は便利だった。。。)

TEST_DATA_SIZE = 100000  # テストデータのサイズ
MINIBATCH_SIZE = 1000  # ミニバッチサイズ
MINIBATCH_SIZE = 256  # ミニバッチサイズ
EVALUATION_SIZE = 1000  # 評価のときのデータサイズ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(train_x, train_y, model, optimizer,loss_func, epoch, ITERATION_NUM, MINIBATCH_SIZE):
    # model.train()
    #for i in range(ITERATION_NUM):
    for i in range(0, train_x.shape[0], MINIBATCH_SIZE):
        ## 学習: 重みのアップデート
        # 学習データをランダムに抽出
        # np.random.choice(a, n, replace=False): 0以上a以下の値からn個の値をランダムに抽出(重複無し)
        index = np.random.choice(train_x.shape[0], MINIBATCH_SIZE, replace=False)
        # 以下で３次元→４次元にreshapeしてる, 微分も有効化
        # x = torch.tensor(train_x[index].reshape(MINIBATCH_SIZE, 1, 8, 8).astype(np.float32), requires_grad=True)
        x = train_x[index]  # size: (MINIBATCH_SIZE, 1, 8, 8)
        t = train_y[index]
        # GPU使用の為の設定。なかったらCPUを用いる。
        x = x.to(device)
        t = t.to(device)
        # 損失関数の設定
        x = model(x)  # (MINIBATCH SIZE, 64) = (100, 64)の形で返ってくる
        train_loss = loss_func(x, t)
        # 勾配の初期化
        optimizer.zero_grad()
        # 勾配の計算
        train_loss.backward()  # back propagationにより勾配算出
        ## 重みをファイルに保存
        if i == int(ITERATION_NUM-1):
            with open('train_loss.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(float(train_loss)) + '\n')
        # 重みの更新
        optimizer.step()

def test(test_x, test_y, model, optimizer, loss_func ,epoch, EVALUATION_SIZE):
    ## 評価
    # model.eval()
    index = np.random.choice(test_x.shape[0], EVALUATION_SIZE, replace=False)
    # 以下で３次元→４次元にreshapeしてる, 微分も有効化
    # x = torch.tensor(test_x[index].reshape(EVALUATION_SIZE, 1, 8, 8).astype(np.float32))
    x = test_x[index]  # size: (MINIBATCH_SIZE, 1, 8, 8)
    t = test_y[index]
    # GPU使用の為の設定。なかったらCPUを用いる。
    x = x.to(device)
    t = t.to(device)
    # 損失関数の設定
    outputs = model(x)  # model.forward(x)の出力値
    validation_loss = loss_func(outputs, t)
    # 正答率の算出
    _, predicted = torch.max(outputs.data, 1)
    validation_acc = predicted.eq(t).sum().numpy() / len(predicted)
    print('epoch: ', epoch, 'val loss: ', float(validation_loss), 'val acc: ', float(validation_acc))
    with open('validation_loss.csv', 'a') as f:
        f.write(str(epoch) + ',' + str(float(validation_loss)) + ',' + str(float(validation_acc)) + '\n')

## データの読み込み
if os.path.isfile('states.npy') and os.path.isfile('actions.npy'):
    states = np.load('states.npy')
    actions = np.load('actions.npy')
else:
    downloaded_dir_path = './data/'
    isExists = os.path.exists(downloaded_dir_path)
    if not isExists:  # 棋譜データのダウンロードがまだの時、ダウンロードを行う
        load.download()
    total_matchs = load.match_counter()
    states, actions = load.load_and_save(total_matchs)  # データの読み込み・加工・保存

## 入力の正規化[0,1]
states = states / 2
"""
states[i]=
[[0.  1.  1.  1.  1.  1.  1.  0. ]
 [0.  1.  1.  1.  1.  1.  0.  0. ]
 [0.5 0.  1.  1.  1.  1.  1.  1. ]
 [0.  0.5 0.5 1.  0.5 0.5 0.5 1. ]
 [0.  1.  0.5 1.  1.  0.5 1.  1. ]
 [1.  1.  1.  0.5 0.5 1.  1.  1. ]
 [0.  0.  0.5 0.5 0.5 0.5 0.  0. ]
 [0.  0.5 0.5 0.5 0.5 0.5 0.  0. ]]
のような形になっていれば良い
"""

## csvファイルが存在する場合一旦削除
val_loss_csv_isExists = os.path.exists('validation_loss.csv')
if val_loss_csv_isExists:
    os.remove('validation_loss.csv')
tr_loss_csv_isExists = os.path.exists('train_loss.csv')
if tr_loss_csv_isExists:
    os.remove('train_loss.csv')

## 訓練用データ(x)の抽出
test_x = states[:TEST_DATA_SIZE].copy()  # load.pyにてランダムに並び替え済み
train_x = states[TEST_DATA_SIZE:].copy()
del states  # メモリがもったいないので強制解放
print('Num of train data: ', len(train_x))
print('Num of test data: ', len(test_x))
## 教師データ(y)の抽出
test_y = actions[:TEST_DATA_SIZE].copy()
train_y = actions[TEST_DATA_SIZE:].copy()
del actions
print('Num of targets for train data: ', len(train_y))
print('Num of targets for test data: ', len(test_y))

## データの整形・Tensor形式に変換
# モデル用にデータを整形(データ数, チャンネル数, height, width)
train_x = train_x.reshape(train_x.shape[0], 1, 8, 8)
test_x = test_x.reshape(test_x.shape[0], 1, 8 ,8)
# PyTorchのテンソルに変換
train_x = torch.Tensor(train_x)
test_x = torch.Tensor(test_x)
train_y = torch.LongTensor(train_y) # pytorchのラベルデータはLong(int64)化整数で扱わないとエラー
test_y = torch.LongTensor(test_y)

## model(ネットワーク構造)の呼び出し
# network.pyにて定義
model = network.AgentNet()
print(model)

## 既に学習済のモデル(model.pth)が存在する場合はこれをmodelインスタンスとして用いる
# if os.path.isfile('model.pth'):
#     model.load_state_dict(torch.load("model.pth"))

## 重み更新則の定義(ADAM)
optimizer = optim.Adam(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr=0.001)

## 使用する損失関数の定義
# x[ラベル値]のときに使える
# -log(softmax)のような形状の関数となるため[0,1]出力とはならない
# x[あるラベル値]が出力される確率に対する`情報エントロピー`
loss_func = nn.CrossEntropyLoss()

# ITERATION_NUM = int(train_x.shape[0] / MINIBATCH_SIZE)  # 全データ数を均等分割するように
## 訓練とテストの実行
EPOCHS = 100
ITERATION_NUM = 100
for epoch in range(EPOCHS):
    ## 訓練用関数呼び出し
    train(train_x, train_y, model, optimizer, loss_func, epoch, ITERATION＿NUM, MINIBATCH_SIZE)
    # モデルを保存する。
    torch.save(model.state_dict(), "model.pth")
    ## テスト用関数呼び出し
    test(test_x, test_y, model, optimizer, loss_func ,epoch, EVALUATION_SIZE)

print("The model has been saved.")
print("Excute game.py to play.")

# from torch.utils.data import TensorDataset, DataLoader
# # 入力（x）とラベル（y）を組み合わせて最終的なデータを作成
# ds_train = TensorDataset(x_train, y_train)
# ds_test = TensorDataset(x_test, y_test)

# # DataLoaderを作成
# loader_train = DataLoader(ds_train, batch_size=MINIBATCH_SIZE, shuffle=True)
# loader_test = DataLoader(ds_test, batch_size=MINIBATCH_SIZE, shuffle=False)

# 学習用関数
# def train(loader_train, model_obj, optimizer, loss_func, device, total_epoch, epoch):
#     model_obj.train() # モデルを学習モードに変更

#     # ミニバッチごとに学習
#     for data, targets in loader_train:
#         # data = data.to(device) # GPUを使用するため，to()で明示的に指定
#         # targets = targets.to(device) # 同上
#         optimizer.zero_grad() # 勾配を初期化
#         outputs = model_obj(data) # 順伝播の計算
#         loss = loss_func(outputs, targets) # 誤差を計算

#         loss.backward() # 誤差を逆伝播させる
#         optimizer.step() # 重みを更新する

#     print ('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))


# テスト用関数
# def test(loader_test, trained_model, loss_func, device, EPOCHS, epoch):
#     trained_model.eval() # モデルを推論モードに変更
#     correct = 0 # 正解率計算用の変数を宣言

#     # ミニバッチごとに推論
#     with torch.no_grad(): # 推論時には勾配は不要
#         for data, targets in loader_test:
#             # data = data.to(device) #  GPUを使用するため，to()で明示的に指定
#             # targets = targets.to(device) # 同上
#             # 損失関数の設定
#             outputs = trained_model(data) # 順伝播の計算
#             validation_loss = loss_func(outputs, targets)
#             # 推論結果の取得と正誤判定
#             _, predicted = torch.max(outputs.data, 1) # 確率が最大のラベルを取得
#             correct += predicted.eq(targets.data.view_as(predicted)).sum() # 正解ならば正解数をカウントアップ

#     # 正解率を計算
#     data_num = len(predicted) # テストデータの総数
#     print ('Epoch [%d/%d], Loss: %.4f' % (epoch, EPOCHS, validation_loss.item()), '\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))

# device = None
# for epoch in range(EPOCHS):
#     train(loader_train, model, optimizer, loss_func, device, EPOCHS, epoch)
#     test(loader_test, model, loss_func, device, EPOCHS, epoch)
