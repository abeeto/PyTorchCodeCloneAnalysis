from numpy import mean,std
from matplotlib import pyplot
import torchvision
import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/',train=True,download=True,transform=transform),
    batch_size=64,
    shuffle=True
)

test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/',train=False,download=True,transform=transform),
    batch_size=1000,
    shuffle=True
)

examples=enumerate(test_loader)
batch_idx,(example_data,example_targets)=next(examples)
print(example_data.shape)

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,5)
        self.conv2=nn.Conv2d(10,20,5)
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)
        

    def forward(self,x):
        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=F.relu(F.max_pool2d(self.conv2(x),2))
        x=x.view(-1,320)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x) 




net=Net()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.5)
train_losses=[]
train_counter=[]
test_losses=[]
test_counter=[i*len(train_loader.dataset) for i in range(n_epochs+1)]


def train(epoch):
    net.train()
    for batch_idx,(X,y) in enumerate(train_loader):
        optimizer.zero_grad()
        output=net(X)
        loss=F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_idx * len(X), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(net.state_dict(), '/model.pth')
            torch.save(optimizer.state_dict(), '/optimizer.pth')



def test():
    net.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for X,y in test_loader:
            output=net(X)
            test_loss+=F.nll_loss(output,y,size_average=False).item()
            pred=output.data.max(1,keepdim=True)[1]
            
            correct+=pred.eq(y.data.view_as(pred)).sum()
    test_loss/=len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1,n_epochs+1):
    train(epoch)
    test()