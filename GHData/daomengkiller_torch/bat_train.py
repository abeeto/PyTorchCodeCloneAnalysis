import torch
import torch.utils.data as Data

BATCH_SIZE = 5
x = torch.linspace(1, 10, 10)  # 从1到10，生成10个数
y = torch.linspace(10, 1, 10)
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)  # 转化为标准的数据格式
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)#通过批处理的方法，生成各种数据批
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch:', epoch, '| Step:', step, '|batch x:', batch_x.numpy(), '|batch y:', batch_y.numpy())
