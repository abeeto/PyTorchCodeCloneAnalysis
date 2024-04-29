'''
2018-09-15
批训练设置，用loader
'''

import torch
import torch.utils.data as Data

BATCH_SIZE=5

x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, #打乱数据顺序
    num_workers=2,  #用两个线程提取
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        #training
        print('Epoch:',epoch,'| step:',step,'| batch x:',batch_x.numpy(),'| batch y:',batch_y.numpy())
