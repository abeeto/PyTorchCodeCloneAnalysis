import torch.nn as nn   #give things like oop
import torch.nn.functional as F    #give things like functions

# initizling nn module by inheriting
class Net(nn.Module):
  def __init__(self):
    super().__init__() 
    self.fc1 = nn.Linear(28*28, 64)
    self.fc2 = nn.Linear(64, 64)
    # we can also put if else here
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10)

  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return F.log_softmax(x, dim=1)


net = Net()
print(net)

#passing data
#mimicking image

X = torch.rand((28,28))
print(X)

#we have to change the view first
X=X.view(-1,28*28) #-1 says that no batch just take the data

output = net(X)
print(output)
