# -*- coding:utf-8 -*-
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # 为CPU设置随机种子，使得每次运行时的随机结果都是相同的

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

# 将数据转为dataset对象
torch_dataset = Data.TensorDataset(x, y)

# 构造数据批处理对象
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,               # 是否打乱数据
    num_workers=2,              # 多进程读取数据
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
