import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

test_dir='testdata/'

train_data=datasets.MNIST(test_dir,train=True,download=True,transform=transforms.ToTensor())
# 为了方便，使用测试集当作验证集
valid_data=datasets.MNIST(test_dir,train=False,download=True,transform=transforms.ToTensor())
train_data=DataLoader(train_data,batch_size=32)
vaild_data=DataLoader(valid_data,batch_size=32)

class SimpleNet(torch.nn.Module):
    def __init__(self,):
        super(SimpleNet,self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(28*28,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,10)
        )
    
    def forward(self,x):
        assert isinstance(x,torch.Tensor)
        return self.net(x.view(-1,28*28))

loss=torch.nn.CrossEntropyLoss()

def metric_acc(y_pred,y_true):
    assert isinstance(y_pred,torch.Tensor)
    assert isinstance(y_true,torch.Tensor)
    y_pred=torch.argmax(y_pred,dim=1)
    acc=torch.eq(y_pred,y_true).float().mean().item()
    return acc

from torcher import Torcher

model=SimpleNet().cuda()
trainer=Torcher(model,loss,opti=lambda x:torch.optim.Adam(x),metrics=metric_acc)
trainer.fit(train_data,valid_data=vaild_data,model_path=test_dir+'dnn4mnist',epochs=10,log_file=test_dir+'log.txt')