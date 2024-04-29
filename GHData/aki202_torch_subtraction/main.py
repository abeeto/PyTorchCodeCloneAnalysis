# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import data_generator
from encoder import Encoder
from decoder import Decoder
import util

char2id = util.get_char2id()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = 200 # 文字の埋め込も次元数
hidden_dim = 128 # LSTMの隠れ層のサイズ
vocab_size = len(char2id) # 扱う文字の数。今回は13文字

#inputs, outputs = data_generator.prepare_data()
#input_tensor    = torch.tensor(inputs[0:2])
#output_tensor   = torch.tensor(outputs[0:2])

encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

#h_n, c_n = encoder(input_tensor)
# h_n: [1, 2, 128]
# c_n: [1, 2, 128]
#print(model)

input_data, output_data = data_generator.prepare_data()
# input_data = (50000, 7)
# output_data = (50000, 5)

train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size= 0.7)
# train_x = (35000, 7)
# test_x  = (35000, 5)
# train_y = (15000, 7)
# test_y  = (15000, 5)

# 学習

BATCH_NUM = 100
EPOCH_NUM = 100

#breakpoint()

all_losses = []
print('training ...')
for epoch in range(1, EPOCH_NUM+1):
  epoch_loss = 0 # epoch毎のloss

  # データをミニバッチに分ける
  input_batch, output_batch = data_generator.train2batch(
    train_x, train_y, batch_size=BATCH_NUM)
  # input_batch  = (350, 100, 7)
  # output_batch = (350, 100, 5)

  for i in range(len(input_batch)):
    # 勾配の初期化
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # データをテンソルに変換
    input_tensor  = torch.tensor(input_batch[i], device=device)  # (100, 7)
    output_tensor = torch.tensor(output_batch[i], device=device) # (100, 5)

    # Encodeの順伝搬
    encode_state = encoder(input_tensor) # (2, 1, 100, 128)

    # Decoderで使うデータはoutput_tensorを１つずらしたものを使う
    # Decoderのインプットとするデータ
    source = output_tensor[:, :-1]

    # Decoderの教師データ
    # 生成開始を表す"_"を削っている
    target = output_tensor[:, 1:]

    loss = 0
    # 学習時はDecoderはこのように１回呼び出すだけでグルっと系列をループしているからこれでOK
    # Sourceが４文字なので、以下のLSTMが４回再帰的な処理をしている
    decoder_output, _ = decoder(source, encode_state)
    # decoder_output.size() = (100, 4, 13)
    # 「13」は生成すべき対象の文字が13文字あるから。decoder_outputの３要素目は
    # [-14.6240, -3.7612, -11.0775, ..., -5.7391, -15.2319, -8.6547]
    # こんな感じの値が入っており、これの最大値に対応するインデックスを予測文字

    for j in range(decoder_output.size()[1]):
      # バッチ毎にまとめてloss計算
      # 生成する文字は４文字なので、４回ループ
      loss += criterion(decoder_output[:, j, :], target[:, j])

    epoch_loss += loss.item()

    # 誤差逆伝播法
    loss.backward()

    # パラメータ更新
    # Encoder, Decoder両方学習
    encoder_optimizer.step()
    decoder_optimizer.step()

  # 損失を標示
  print('Epoch %d: %.2f' % (epoch, epoch_loss))
  all_losses.append(epoch_loss)
  if epoch_loss < 1: break

print('Done')
