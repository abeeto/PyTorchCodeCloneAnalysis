import os, requests, json
from datetime import datetime as dt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from slackbot import SlackMessenger

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import GoogleCloudStorageObserver



NAME = 'MNIST_demo_'+dt.now().isoformat()[0:16]

slack_credentials = json.load(open(os.path.join(os.getcwd(),'credentials','slack.json'),'r'))
gcp_conf = json.load(open(os.path.join(os.getcwd(),'credentials','gcp_bucket.json'),'r'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(),'credentials','gcp_credentials.json')

ex = Experiment(NAME)
ex.observers.append(FileStorageObserver('experiments'))
ex.observers.append(GoogleCloudStorageObserver(bucket=gcp_conf['bucket'], basedir=gcp_conf['basedir']))

@ex.config
def config():
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    slack_token = slack_credentials['token']
    target= slack_credentials['target']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

@ex.automain
def main(n_epochs, batch_size_train, batch_size_test, learning_rate, momentum, log_interval, slack_token, target):
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(os.path.join(os.getcwd(),'data'), train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(os.path.join(os.getcwd(),'data'), train=False, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_test, shuffle=True)
    
    
    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                  output = network(data)
                  test_loss += F.nll_loss(output, target, size_average=False).item()
                  pred = output.data.max(1, keepdim=True)[1]
                  correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

            return test_loss, float(int(correct)/len(test_loader.dataset))

    def train(epoch,verbose=False, callbacks = []):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                if verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

        # on epoch end
        test_loss, test_acc = test()
        ex.log_scalar('test_loss',test_loss,epoch)
        ex.log_scalar('test_acc',test_acc,epoch)
        if verbose:
            print (f'Epoch: {epoch}; Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

        for callback in callbacks:
            callback.on_epoch_end(epoch,test_loss, test_acc)
        torch.save(network.state_dict(), os.path.join(os.getcwd(),'tmp','model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(os.getcwd(),'tmp','optimizer.pth'))
    

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    slack_callback = SlackMessenger(slack_token,target,'hello_MNIST')
    
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch, verbose=True, callbacks=[slack_callback])
        
        
    ex.add_artifact(filename=os.path.join(os.getcwd(),'tmp','model.pth'), name='saved_model.pth')
    ex.add_artifact(filename=os.path.join(os.getcwd(),'tmp','optimizer.pth'), name='saved_optimizer.pth')
    
        
if __name__=="__main__":
    main()