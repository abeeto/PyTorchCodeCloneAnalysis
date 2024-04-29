import torch
import numpy as np
#DataSet是抽象类，无法实例化,只能由自定义的类去继承
from torch.utils.data import Dataset
#DataLoader可实例化
from torch.utils.data import DataLoader

# 自己定义一个类，表示继承于Dataset
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        # 读取数据集，“,”作为分隔符，读取32位浮点数
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        #获得数据集长度
        self.len=xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
    #获得索引方法
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    #获得数据集(datalength)长度
    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv.gz')
#dataset数据集 batch_size多少个为一个数据集 shuffle是否打乱 num_workers表示多线程的读取
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ =='__main__':
    for epoch in range(100):
        #enumerate:可获得当前迭代的次数
        for i,data in enumerate(train_loader,0):
            #准备数据dataloader会将按batch_size返回的数据整合成矩阵加载
            inputs, labels = data
            #前馈
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            #更新
            optimizer.step()