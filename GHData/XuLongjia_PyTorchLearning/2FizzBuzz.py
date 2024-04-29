def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i %5 == 0: return 2
    elif i %3 == 0: return 1
    else: return 0

def fizz_buzz_decode(i,prediction):
    return[str(i),"fizz",'buzz','fizzbuzz'][prediction]

# for i in range(1,16):
#    print(fizz_buzz_decode(i,fizz_buzz_encode(i)))

#定义模型的输入与输出
import numpy as np
import torch

NUM_DIGITS = 10
def binary_encode(i,num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

trX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2**10)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**10)])

#用Pytorch定义模型
NUM_HIDDEN=100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN,4)
)

if torch.cuda.is_available():
    model = model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)

#开始训练
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred,batchY)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    loss = loss_fn(model(trX),trY).item()
    print('Epoch', epoch,"Loss", loss)