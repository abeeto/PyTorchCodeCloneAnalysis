# coding: utf-8
import network  # network.pyで定義されているクラスをimport
import numpy as np
import os, sys

import torch
import torch.nn.functional as F
import torch.nn as nn

## ボード情報
BOARD_SIZE = 8
CENTER_IDX = int(BOARD_SIZE / 2)

## 手の表現 x軸をアルファベットで、y軸を1~8の数値で表現する
IN_ALPHABET = [chr(i) for i in range(97,97+BOARD_SIZE)]  # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# IN_NUMBER = [str(i) for i in range(1, BOARD_SIZE + 1)]  # ['1', '2', '3', '4', '5', '6', '7', '8']
IN_NUMBER = [str(i) for i in range(BOARD_SIZE)]  # ['0', '1', '2', '3', '4', '5', '6', '7']


## 入力された手の形式をチェック
def checkIN(IN):
    # INが空でないかチェック
    if not IN:
        return False

    # Inの１文字目と２文字目がそれぞれa~h, 0~7の範囲内であるかチェック
    if IN[0] in IN_ALPHABET:
        if IN[1] in IN_NUMBER:
            return True

    return False

# 1�^�[���V�~�����[�g
# position  y, x
# color  1:��, 2:��
# return  True:�L��, False:����
def simulation(state, position, color):
    # y = position[0] - 1
    y = position[0]
    x = position[1]

    # 0(空白)でないところには打てない
    if state[y, x] != 0:
        return False

    # 8�����{��
    is_changed = False  # �Ђ�����Ԃ��ꂽ���ǂ���
    for i in range(3):  # i - 1 = -1, 0, 1�ƂȂ�ړ��������\����
        for j in range(3):
            if i - 1 == 0 and j - 1 == 0:  # ����0���ƈړ����Ȃ�
                continue
            for k in range(1, 8):
                xt = x + k * (i - 1)
                yt = y + k * (j - 1)
                if not (0 <= xt <= 7 and 0 <= yt <= 7):
                    break
                elif state[yt, xt] == 0:
                    break
                elif state[yt, xt] == color:
                    if k > 1:
                        for l in range(0, k):
                            xt_ = x + l * (i - 1)
                            yt_ = y + l * (j - 1)
                            state[yt_, xt_] = color
                        is_changed = True
                    break

    return is_changed


# �p�X�m�F
# color  1:��, 2:��
# return  True:�p�X, False:�p�X�Ȃ�
def is_pass(state, color):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if simulation(state.copy(), (i, j), color):
                return False
    return True


## マップの表示系
def show(state):
    print(' ', end='')

    ## 横軸部表示
    horizontal_alphabet = ' | '.join(IN_ALPHABET)  # listの要素間を半角空白をもって結合
    print(' | ' + horizontal_alphabet + ' |')
    print('-----------------------------------')

    ## 縦軸部とマップ記号を表示
    for i in range(BOARD_SIZE):
        print(i, '|',end='')
        for j in range(BOARD_SIZE):
            if state[i][j] == 0:
                print(' □ |', end='')
            elif state[i][j] == 1:
                print(' ○ |', end='')
            elif state[i][j] == 2:
                print(' ● |', end='')
        print()  # ���s
        print('-----------------------------------')

# if __name__ == '__main__':
#     state = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
#     state[CENTER_IDX, CENTER_IDX-1] = 1
#     state[CENTER_IDX-1, CENTER_IDX] = 1
#     state[CENTER_IDX-1, CENTER_IDX-1] = 2
#     state[CENTER_IDX, CENTER_IDX] = 2
#     show(state)

## モデルを外部ファイルとして読み込む（その前に初期化する必要あり）
model = network.AgentNet()

## パラメータの読み込み
if os.path.isfile('./model.pth'):
    # 既に学習済のモデル(model.pth)が存在する場合はこれをmodelインスタンスとして用いる
    params = torch.load("model.pth")
    model.load_state_dict(params)
    model.load_state_dict(torch.load("model.pth"))
else:
    # 学習済みモデルがないときはtrain.pyの実行を促すようにする
    print("No trained model.")
    print("Excute train.py with \'python3 train.py\' to get the trained model.")  # theいるよね？
    sys.exit( )

# y, x
## マップの初期状態を生成
# 0: 石の置き場, 1: 黒, 2: 白
state = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
state[CENTER_IDX, CENTER_IDX-1] = 1
state[CENTER_IDX-1, CENTER_IDX] = 1
state[CENTER_IDX-1, CENTER_IDX-1] = 2
state[CENTER_IDX, CENTER_IDX] = 2

print('You are white, second.')  # �R���s���[�^�[����1�ɂȂ�悤��
print("Input: Coordination formatted such as \'f5\'")
print()  # ���s

former_is_pass = False  # �����p�X�ŏI���̂��߁A�O��p�X���������L�^

show(state)  # ��Ղ̏�Ԃ�\��
print()

## メインの対戦処理, AI側からスタート
while True:
    # �R���s���[�^�[��
    if is_pass(state, 1):
        if former_is_pass:
            break
        else:
            former_is_pass = True
    else:
        former_is_pass = False

        state_var = torch.tensor(state.reshape(1, 1, 8, 8).astype(np.float32))
        model_output = model(state_var)
        action_probabilities = F.softmax(model_output, dim=1) # 確率値が付与された碁盤
        # action_probabilities = model.predictor(state_var).data.reshape(64)

        # �m�����ɍs������ׂ�
        # �K���ɏ������̂ŃA���S���Y���̌����������Ǝv��
        action_list = [0]  # ���X�g
        for i in range(1, 64):
            for j in range(i):
                if action_probabilities[0][i] > action_probabilities[0][action_list[j]]:
                    action_list.insert(j, i)
                    break
            else:
                action_list.append(i)

        for i in range(64):
            action = action_list[i]
            position = (action // 8, action % 8)  # (y, x)のタプル形式となる
            alphabeted_x_pos = IN_ALPHABET[position[1]]  # x座標(position[1])をアルファベットに変換
            # print('{0},{1}'.format(position[0], position[1]), end=' ')
            print('{0}{1}'.format(alphabeted_x_pos, position[0]), end=' ')
            if simulation(state, position, 1):
                break
        print()  # ���s
        print()  # ���s
        print("++++++++++++++++++++++++++")
        print()  # ���s
        # print(state)
        # print(action_list)
        show(state)  # ��Ղ̏�Ԃ�\��
        print()  # ���s
        print()  # ���s

    # ���[�U�[��
    if is_pass(state, 2):
        if former_is_pass:
            break
        else:
            former_is_pass = True
    else:
        former_is_pass = False

        while True:
            print("You\'re turn.")
            IN = input("Input: ")
            # 手入力をチェック
            if checkIN(IN):
                x = IN_ALPHABET.index(IN[0])
                # y = IN_NUMBER.index(IN[1]) + 2
                y = IN_NUMBER.index(IN[1])
                position = [y, x]
                print(position)
            else:
                print("Input the coordination in RIGHT format (ex. f5).")
                continue

            # position = [int(e) for e in input().split(',')]  # 入力受付部
            if simulation(state, position, 2):
                break
            else:
                print("You CANNOT place there.")
        print("------------------------------")
        print()  # ���s

        show(state)  # ��Ղ̏�Ԃ�\��
        print()  # ���s


## スコア計算
black_score = 0
white_score = 0
for i in range(8):
    for j in range(8):
        if state[i][j] == 1:
            black_score += 1
        elif state[i][j] == 2:
            white_score += 1


## 結果表示
if black_score == white_score:
    print('Draw.  b(computer) - w(you) :', black_score, '-', white_score)
elif black_score > white_score:
    print('Computer Win.  b(computer) - w(you) :', black_score, '-', white_score)
else:
    print('You Win!  b(computer) - w(you) :', black_score, '-', white_score)
