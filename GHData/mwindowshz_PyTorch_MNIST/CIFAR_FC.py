# code based on https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms

'''class for out network, to classifie MNIST digits
we have 4 fc layers.
1 - 28*28 image input, second and third fc of 200, and 4th is 10 for scores for each digit class '''
class Net(nn.Module):
 
    def __init__(self):
        ''' class constructor 
        we call super() to instatiate the parent class nn.Module'''
        super(Net,self).__init__()
        self.fc1 = nn.Linear(32*32*3,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,100)

    def forward(self,x):
        '''forwoard() method needs to be implemeted in child class according to our specific network.\t
        x - is the image input, then it is passed to layer fc1, then results are passed to fc2, up to fc3
        we chose activation for fc1,fc2
        last layer we use we return a log softmax “activation”. 
        This, combined with the negative log likelihood loss function which will be defined later, gives us a multi-class cross entropy based loss function which we will use to train the network.'''

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x= self.fc3(x)
        return F.log_softmax(x)

image_size = 32
net = Net()
print(net)

#hyperparametrers
learning_rate = 0.01
epochs = 15
logging_interval = 10
batch_size = 200


train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data_CIFAR', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data_CIFAR', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)


#train the network
# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9)
#create loss function - negative log likelihood loss
criterion = nn.NLLLoss()  #this combined with our log softmax output from the neural network gives us an equivalent cross entropy loss for our 10 classification classes.

#run main training loop
for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = Variable(data),Variable(target)
        #resize data from (batchsize,1,28,28) to (batch_size,28*28) - the .view() function operates on PyTorch variables to reshape them
        data = data.view(-1,32*32*3)  #the “-1” notation in the size definition. So by using data.view(-1, 28*28) we say that the second dimension must be equal to 28 x 28, but the first dimension should be calculated from the size of the original data variable ,  In practice, this means that data will now be of size (batch_size, 784). We can pass a batch of input data like this into our network
        # we zero all gradients to prepare for next backpropegations pass
        optimizer.zero_grad()
        #pass the input data batch into the model 
        net_out = net(data)
        loss = criterion(net_out,target) #calculate loss
        loss.backward()    # runs a back-propagation operation from the loss Variable backwards through the network
        optimizer.step()   # we tell PyTorch to execute a gradient descent step based on the gradients calculated during the .backward() operation.
        if batch_idx % logging_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))

# run a test loop
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    data = data.view(-1, 32 * 32*3)
    net_out = net(data)
    # sum up batch loss
    test_loss += criterion(net_out, target).data
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


#save the model
import os
dirpath = os.getcwd()
model_name = dirpath + 'model-cifar.torch'
torch.save(net.state_dict,model_name)

# labels = unpickle('../data_CIFAR/batches.meta')[b'label_names']
lable = test_loader.dataset.test_labels[1]
imgData = test_loader.dataset.test_data[1]

from matplotlib import pyplot as plt
plt.imshow(imgData, interpolation='bicubic')
plt.show()
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1
