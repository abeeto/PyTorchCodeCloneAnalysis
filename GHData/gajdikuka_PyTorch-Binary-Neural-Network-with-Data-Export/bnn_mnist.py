from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import BinarizeLinear, BinarizeSign
from export_modules import SaveModel, SaveInput

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=256, metavar='N')
                    
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
                    
parser.add_argument('--epochs', type=int, default=10, metavar='N')
                    
parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
                    
parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
                    
parser.add_argument('--no-cuda', action='store_true', default=False)
                    
parser.add_argument('--seed', type=int, default=1, metavar='S')
                    
parser.add_argument('--gpus', default=3)

parser.add_argument('--log-interval', type=int, default=10, metavar='N')

parser.add_argument('--export-model-file', default='output/modelSave', metavar='N')

parser.add_argument('--export-input-file', default='output/inputSave', metavar='N')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(784)
        self.bin0 = BinarizeSign()
        
        self.fc1 = BinarizeLinear(784, 200)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200)
        self.bin1 = BinarizeSign()
        
        self.fc2 = BinarizeLinear(200, 100)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100)
        self.bin2 = BinarizeSign()
        
        self.fc3 = BinarizeLinear(100, 100)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(100)
        self.bin3 = BinarizeSign()
        
        self.fc4 = BinarizeLinear(100, 100)
        self.htanh4 = nn.Hardtanh()
        self.bn4 = nn.BatchNorm1d(100)
        self.bin4 = BinarizeSign()
        
        self.fc5 = BinarizeLinear(100, 10)
        self.logsoftmax=nn.LogSoftmax(1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.bn0(x)
        x = self.bin0(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.bin1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.bin2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.bin3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.htanh4(x)
        x = self.bin4(x)
        
        x = self.fc5(x)
        return self.logsoftmax(x)
    
model = Net()
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(np.multiply(data.cpu(), 255).cuda()), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(np.multiply(data.cpu(), 255).cuda()), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
   train(epoch)
   test()
   
print("Saving model")
SaveModel(model, args.export_model_file)

print("Saving inputs")
SaveInput(model, args.export_input_file)