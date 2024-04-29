
import torch
import torch.nn as nn

##搭建网络

class Cifar(nn.Module):
    def __init__(self):
        super(Cifar,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,2)
        )

    def forward(self,x):
        x=self.model(x)
        return x

if __name__=='__main__':
    model=Cifar()
    input=torch.ones((64,3,32,32))
    output=model(input)
    print(output.shape)
    print(output)
    
