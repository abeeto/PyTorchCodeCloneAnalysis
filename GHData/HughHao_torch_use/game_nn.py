# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:45
# @Author : hhq
# @File : game_nn.py
# todo FizzBuzz是一个简单的小游戏。游戏规则如下：从1开始往上数数，当遇到3的倍数的时候，说fizz，当遇到5的倍数，说buzz，当遇到15的倍数，
#  就说fizzbuzz，其他情况下则正常数数。
# One-hot encode the desired outputs: [number, \"fizz\", \"buzz\", \"fizzbuzz\"],
import numpy as np
import torch

use_gpu = torch.cuda.is_available()


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


# 输出列表中的字符，根据i和prediction


##我们首先定义模型的输入与输出(训练数据)
NUM_DIGITS = 10


# todo 判断奇偶
# for i in range(100):  # 打印所有奇数
#     if i & 1 == 1:
#         print(i)
# todo 消去最后一位
# x & (x - 1)
# todo & 运算前将所有数字转化为二进制，然后对照计算
# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


# i >> d表示将数字i转化为二进制并且向右移位d个所得数值和1做与运算
# 将数字1表示为二进制，二进制最大为2 ** NUM_DIGITS-1


trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])  # 目标值
# 然后我们用PyTorch定义模型
# Define the model
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)

# "- 为了让我们的模型学会FizzBuzz这个游戏，我们需要定义一个损失函数，和一个优化算法。\n",
#     "- 这个优化算法会不断优化（降低）损失函数，使得模型的在该任务上取得尽可能低的损失值。\n",
#     "- 损失值低往往表示我们的模型表现好，损失值高表示我们的模型表现差。\n",
#     "- 由于FizzBuzz游戏本质上是一个分类问题，我们选用Cross Entropyy Loss函数。\n",
#     "- 优化函数我们选用Stochastic Gradient Descent。"
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# todo 使用GPU和CPU
if use_gpu:
    model = model.cuda()
    loss_fn = loss_fn.cuda()
    trX = trX.cuda()
    trY = trY.cuda()
# Start training it\n",
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):  # BATCH_SIZE小于len(trX),start取值0, len(trX)之间的BATCH_SIZE的倍数
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
        optimizer.zero_grad()
        # loss = loss.cpu()
        loss.backward()  # 利用一批样本减小损失训练
        optimizer.step()
        # Find loss on training data
        loss = loss_fn(model(trX), trY).item()  # item()的作用是取出单元素张量的元素值并返回该值，保持该元素类型不变。
        # 输出全部数据的损失
        print('Epoch:', epoch, 'Loss:', loss)

##最后我们用训练好的模型尝试在1到100这些数字上玩FizzBuzz游戏
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])  # 转化为tensor
if use_gpu:
    testX = testX.cuda()
with torch.no_grad():
    # model = model.cuda()
    testY = model(testX)

predictions = zip(range(1, 101), testY.max(1)[1].data.tolist())
print([fizz_buzz_decode(i, x) for (i, x) in predictions])
testY = testY.cpu()
print(testY)
print(np.sum(testY.numpy().max(1)[1] == np.array([fizz_buzz_encode(i) for i in range(1, 101)])))
