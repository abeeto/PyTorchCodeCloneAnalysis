import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)
# 包装数据和目标张量的数据集
torch_dataset = Data.TensorDataset(x,y)
# 数据加载器 组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
loader =Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    #num_workers=2
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):# enumerate枚举器，返回索引与内容
        # training------------------
        print('Epoch:',epoch,'|Step:',step,'|batch x:',
              batch_x.numpy(),'|batch y:',batch_y.numpy())



