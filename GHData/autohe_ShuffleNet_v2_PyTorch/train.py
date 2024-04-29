import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from ShuffleNetv2 import ShuffleNetV2
from datagen import MyDataset

#set path
rootPath = '/home/ggy/disk1/ggy/code/Adjacent-frames/YouTube_Pose'
txtPath = osp.join(rootPath,'TemporalNet','data','txt')
pathFile = osp.join(txtPath,'path.txt')

testList = osp.join(txtPath,'test.txt')

use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
trainset = MyDataset(path_file=pathFile,list_file=trainList,numJoints = 6,type=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testset = MyDataset(path_file=pathFile,list_file=testList,numJoints = 6,type=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

# Model
net = ShuffleNetV2()
if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=[0,1])
    net.cuda()
    cudnn.benchmark = True

base_lr = 0.000001
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)

#loss
criterion = nn.MSELoss()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx,(input,output) in enumerate(trainloader):
        if use_cuda:
            input = input.cuda()
            output = output.cuda()
        input = Variable(input)
        output = Variable(output)

        optimizer.zero_grad()
        preds = net(input)
        loss = criterion(preds,output)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], train_loss / (batch_idx + 1)))

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx,(input,output) in enumerate(trainloader):
        if use_cuda:
            input = input.cuda()
            output = output.cuda()
        input = Variable(input)
        output = Variable(output)
    preds = net(input)
    loss = criterion(preds, output)
    test_loss += loss.data[0]
    print('%.3f %.3f' % (loss.data[0], test_loss / (batch_idx + 1)))

    # Save checkpoint.
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss

def adjust_learning_rate(optimizer,epoch):
    '''
    the learning rate multiply 0.5 every 50 epoch
    '''
    if epoch%50 ==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    adjust_learning_rate(optimizer,epoch)
    if epoch % 10 == 0:
        test(epoch)