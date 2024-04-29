# * coding:utf-8 *
#@author    :mashagua
#@time      :2019/5/1 10:11
#@File      :lesson_2.py
#@Software  :PyCharm
def fizz_buzz_encode(i):
    if i%15==0:return 3
    if i%3==0:return 2
    if i%5==0:return 1
    else:return 0
def fizz_buzz_decode(i,prediction):
    return [str(i),"fizzbuzz","fizz","buzz"][prediction]

# def helper(i):
#     print(fizz_buzz_decode(i,fizz_buzz_encode(i)))
# for i in range(16):
#     helper(i)

import numpy as np
import torch
NUM_DIGITS=10
def binary_encder(i,num_digits):
    return np.array([i>>d & i for d in range(num_digits)][::-1])
trX=torch.Tensor([binary_encder(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
tryY=torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])
NUM_HIDDED=100
model=torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDED),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDED,4)
)
if torch.cuda.is_available():
    model=model.cuda()
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
BATCH_SIZE=128
for epoch in range(500):
    for start in range(0,len(trX),BATCH_SIZE):
        end=start+BATCH_SIZE
        batchX=trX[start:end]
        batchY=tryY[start:end]
        if torch.cuda.is_available():
            batchX=batchX.cuda()
            batchY=batchY.cuda()
        y_pred=model(batchX)
        loss=loss_fn(y_pred,batchY)
        print("Epoch {},loss {}".format(epoch,loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
testX=torch.Tensor([binary_encder(i,NUM_DIGITS) for i in range(1,101)])
if torch.cuda.is_available():
    testX=testX.cuda()
with torch.no_grad():
    testY=model(testX)
prediction=zip(range(1,101),testY.max(1)[1].cpu().data.tolist())
print([fizz_buzz_decode(i,x) for i,x in prediction])


