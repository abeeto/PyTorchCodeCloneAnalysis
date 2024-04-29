import torch
import torchvision
from torchvision import transforms,datasets
train=datasets.MNIST("", train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))
test=train=datasets.MNIST("", train=False, download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))
trainset= torch.utils.data.DataLoader(train,batch_size=12,shuffle=True)
testset= torch.utils.data.DataLoader(test,batch_size=12,shuffle=True)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784, 64)
        self.fc2=nn.Linear(64, 64)
        self.fc3=nn.Linear(64, 64)
        self.fc4=nn.Linear(64, 10)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return F.log_softmax(x, dim=1)
        return x
net=NeuralNet()


X=torch.rand((28,28))
X=X.view(-1,28*28)
print(net(X))
