import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader,TensorDataset


#***************************************    method first  ***********************************************************
#Dataset是Pytorch中的一个抽象Class，所有的datasets都应该是它的子类，并且应该重写len和getitem来覆盖，
# 其中getitem支持从整数（0,len(dataset)）进行indexing。
#例子：
#我们生成数据集（x,y）其中 y = 5x + xsin(x) + noise。
#代码如下：

#作者：小黑的自我修养
#链接：https://www.jianshu.com/p/3fa75db88387

#定义Mydataset继承自Dataset,并重写__getitem__和__len__
class Mydataset(Dataset):
    def __init__(self, num):
        super(Mydataset, self).__init__()
        self.num = num #生成多少个点（多少个数据）

        def linear_f(x):
            y = 5 * x + np.sin(x) * x + np.random.normal(
                0, scale=1, size=x.size) # y = 5*x + x*sin(x) + noise
            return y

        self.x_train = np.linspace(0, 50, num=self.num) #从0-50生成num多个点
        self.y_train = linear_f(self.x_train)
        self.x_train = torch.Tensor(self.x_train)#转化为张量
        self.y_train = torch.Tensor(self.y_train)
    # indexing
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    #返回数据集大小，应该是（x_transpose,y_transpose）大小即num*2，这里我直接返回了num
    def __len__(self):
        return self.num
if __name__ == "__main__":
    num = 30
    myset = Mydataset(num=num)
    myloader = DataLoader(dataset=myset, batch_size=1, shuffle=False)
    for data in myloader:
       print(data)

#***************************************    method second  ***********************************************************

def linear_f(x):
    y = 5 * x + np.sin(x) * x + np.random.normal(
        0, scale=1, size=x.size) # y = 5*x + x*sin(x) + noise
    return y


if __name__ == "__main__":
    num = 30
    x_train = np.linspace(0, 50, num=num) #从0-50生成num多个点
    y_train = linear_f(x_train)
    x_train = torch.Tensor(x_train)#转化为张量
    y_train = torch.Tensor(y_train)
    myset = TensorDataset(x_train,y_train)
    myloader = DataLoader(dataset=myset, batch_size=1, shuffle=False)
    for data in myloader:
        print(data)