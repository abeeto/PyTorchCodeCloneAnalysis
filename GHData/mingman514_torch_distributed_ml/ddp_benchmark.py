"""
Synchronous SGD training on MNIST
Use distributed MPI backend
PyTorch distributed tutorial:
    http://pytorch.org/tutorials/intermediate/dist_tuto.html
This example make following updates upon the tutorial
1. Add params sync at beginning of each epoch
2. Allreduce gradients across ranks, not averaging
3. Sync the shuffled index during data partition
4. Remove torch.multiprocessing in __main__
"""
import os
import sys
import torch
import torch.utils.data                                                         
import torch.utils.data.distributed 
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np

from math import ceil
#from random import Random
#from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel
from torchsummary import summary
from efficientnet_pytorch import EfficientNet

from timeit import default_timer as timer

gbatch_size = 32
datapath = '/freeflow/shrd_datasets'
MASTER = 0 
TESTING = True

def wrap_resnet_cifar_10(model):
    return nn.Sequential(*[model, nn.ReLU(), nn.Linear(1000, 10)]) 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 60 , kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        # last conv layer 
        # ( [0-9]+ , z,) 
        # AdaptiveAvgPool2d ((x, y)) 
        # linear
        # nn.Linear(x * y * z, [0-9]+ ) 
        self.fc1 = nn.Linear(60 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.pool(x) 
        x = x.flatten() 
        x = x.reshape(-1, 60 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x  


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)  # kernel size, stride, padding = 0 (default)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # input features, output features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_dataset():
    transform = transforms.Compose(
            [transforms.ToTensor(),
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transforms.Normalize((0.1307,), (0.3081,))])

    #dataset = datasets.MNIST(
    dataset = datasets.CIFAR10(
        datapath,
        train=True,
        download=True,
        transform=transform)

    size = dist.get_world_size()
    bsz = int(gbatch_size / float(size))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)       
    train_set = torch.utils.data.DataLoader(                                    
        dataset, batch_size=bsz, shuffle=(train_sampler is None), sampler=train_sampler)
    return train_set, bsz


def load_test_dataset():
    transform = transforms.Compose(
            [transforms.ToTensor(),
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transforms.Normalize((0.1307,), (0.3081,))])

#    dataset = datasets.MNIST(
    dataset = datasets.CIFAR10(
        datapath,
        train=False,
        download=True,
        transform=transform)

    size = dist.get_world_size()
    bsz = int(gbatch_size / float(size))
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)       
    test_set = torch.utils.data.DataLoader(                                    
        dataset, batch_size=bsz, shuffle=(train_sampler is None), sampler=train_sampler)
    return test_set, bsz


def test_accuracy(model, test_set): 
    model.eval()
    test_losses = [] 
    correct_labels = 0 
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1) 
    with torch.no_grad(): 
        for data, target in test_set:
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            test_losses.append(loss.item()) 
            
            label_predict = torch.argmax(softmax(output), dim=-1) 
            correct_labels += torch.sum((target == label_predict)).item()  
    print(f"test_loss {np.mean(test_losses)}")
    print(f"test_accuracy {correct_labels/100}%")
    return test_losses, correct_labels



def init_model():
    # Select model
    model = Net()
    if len(sys.argv) == 1 :
        model = Net()
        print('###### Use default model ######')
        return model

    _model = sys.argv[1]
    if _model == 'lenet5':
        model = Lenet5()
    elif _model == 'resnet18':
        model = wrap_resnet_cifar_10(models.resnet18())
    elif _model == 'resnet34':
        model = models.resnet34()
    elif _model == 'resnet50':
        model = models.resnet50()
    elif _model == 'resnet101':
        model = models.resnet101()
    elif _model == 'resnet152':
        model = models.resnet152()
    elif _model == 'efficientnet':
        model = EfficientNet.from_name('efficientnet-b0')

    else:
        print('###### Incorrect model name ######')
        sys.exit()

    print('###### Use model {} ######'.format(_model))

    return model

def run(rank, size):
    print("RUN CODE STARTS")
    train_set, bsz = load_dataset()
    model = init_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    group = dist.new_group(list(range(size))) # all nodes join learning
    num_batches = ceil(len(train_set.dataset) / (float(bsz) * dist.get_world_size()))
    criterion = nn.CrossEntropyLoss(reduction='mean') 
    # Start training
    for epoch in range(3):
        epoch_loss = 0.0 

        for i, (data, target) in enumerate(train_set):
            # slave compute the forward path
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.data
            loss.backward()

            # aggregates gradients 
            param_list = list(model.parameters())
            
            for j, param in enumerate(model.parameters()):
                dist.reduce(param.grad.data, MASTER, op=dist.ReduceOp.SUM, group=group)
                # Average params in master node
                if rank == MASTER:
                    param.grad.data /= float(size)

                # Broadcast params to slave nodes
            
            if rank == MASTER:
                optimizer.step()
            for param in model.parameters():
                dist.broadcast(param, MASTER, group=group)
  
        print('Rank #{} Epoch {} Loss {:.6f} Global batch size {} on {} ranks'.format(rank,epoch, epoch_loss / num_batches, gbatch_size, dist.get_world_size()))
    return model 
        #    if TESTING:
        #        break
                #sys.exit()

#    summary(model, (1, 28, 28))

if __name__ == "__main__":
    start_t = timer()
    dist.init_process_group(backend='mpi')
    mpi_ready_t = timer()

    size = dist.get_world_size()
    rank = dist.get_rank()
    print('size: {}  rank: {}'.format(size, rank))

    torch.manual_seed(rank)
    np.random.seed(rank)

    train_t = timer()
    model = run(rank, size)
    print('Program End')
    end_t = timer()
    
    if rank == MASTER:
        state_dict = model.state_dict()
        with open("distribute_learned.pkl", "wb") as f:  
            torch.save(state_dict, f)
        runtime = end_t - start_t
        mpitime = mpi_ready_t - start_t
        traintime = end_t - train_t
        
        print('MPI Init : {:.4f}s({:.2f}%)'.format(mpitime, mpitime/runtime*100))
        print('Learning : {:.4f}s({:.2f}%)'.format(traintime, traintime/runtime*100))
        print('  Total  : {:.4f}s({:.2f}%)'.format(runtime, runtime/runtime*100))
