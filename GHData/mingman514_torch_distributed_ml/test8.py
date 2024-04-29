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

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel
from torchsummary import summary

gbatch_size = 32
datapath = '/freeflow/shrd_datasets'
MASTER = 0 


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        """
        Be cautious about index shuffle, this is performed on each rank
        The shuffled index must be unique across all ranks
        Theoretically with the same seed Random() generates the same sequence
        This might not be true in rare cases
        You can add an additional synchronization for 'indexes', just for safety
        Anyway, this won't take too much time
        e.g.
            dist.broadcast(indexes, 0)
        """
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

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


class Net(nn.Module):
    """ Network architecture. """

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
        return F.log_softmax(x, dim=1)


def partition_dataset():
    """ Partitioning MNIST """
    """ 
    transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
    dataset = datasets.MNIST(
        datapath,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    """
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.CIFAR10(
        datapath,
        train=True,
        download=True,
        transform=transform)
#    """
    size = dist.get_world_size()
    bsz = int(gbatch_size / float(size))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)       
    train_set = torch.utils.data.DataLoader(                                    
        dataset, batch_size=bsz, shuffle=(train_sampler is None), sampler=train_sampler)
    return train_set, bsz


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    print("RUN CODE STARTS")
    train_set, bsz = partition_dataset()
#    model = Lenet5()
#    model = Net()
    model = models.resnet152()
    model = DistributedDataParallel(model) # model in which PS sending data does eixt?
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    nodes_in_stake = list(range(size))
    print('nodes_in_stake: ', nodes_in_stake)
    group = dist.new_group(nodes_in_stake)
    num_batches = ceil(len(train_set.dataset) / (float(bsz) * dist.get_world_size()))
    #print("num_batches = ", num_batches) 
    for epoch in range(2):
        epoch_loss = 0.0 
        print('Train set length: ', len(train_set))
        i=0
        
        #"""
        for data, target in train_set:
            if i!=0:
                break
            i+=1
            # slave compute the forward path 
            # i.e., compute the gradient, but do not update 
            data, target = Variable(data), Variable(target)
            # print('data is ', data)
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            model.zero_grad()
            loss.backward()
            """
        for i, data in enumerate(train_set, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            epoch_loss += loss.data
            model.zero_grad()
            loss.backward()
            """
            # aggregates gradients 
            for param in model.parameters():
                dist.reduce(param.grad.data, MASTER, op=dist.ReduceOp.SUM, group=group) 
                # Average params in master node
                if rank == MASTER:
                    param.grad.data /= float(size)
                # Broadcast params to slave nodes
                dist.broadcast(param.grad.data, MASTER, group=group)
               # print('Rank: ',rank, ' BROADCAST Result: ', param.grad.data[0])
                optimizer.step()

#        if rank == MASTER:
            print('Rank #{} Epoch {} Loss {:.6f} Global batch size {} on {} ranks'.format(
                rank,epoch, epoch_loss / num_batches, gbatch_size, dist.get_world_size()))
#    summary(model, (1, 28, 28))
#            # end slave training 

#def init_print(rank, size, debug_print=True):
#    if not debug_print:
#        """ In case run on hundreds of nodes, you may want to mute all the nodes except master """
#        if rank > 0:
#            sys.stdout = open(os.devnull, 'w')
#            sys.stderr = open(os.devnull, 'w')
#    else:
#        # labelled print with info of [rank/size]
#        old_out = sys.stdout
#        class LabeledStdout:
#            def __init__(self, rank, size):
#                self._r = rank
#                self._s = size
#                self.flush = sys.stdout.flush
#
#            def write(self, x):
#                if x == '\n':
#                    old_out.write(x)
#                else:
#                    old_out.write('[%d/%d] %s' % (self._r, self._s, x))
#
#        sys.stdout = LabeledStdout(rank, size)

if __name__ == "__main__":
    dist.init_process_group(backend='mpi')
    size = dist.get_world_size()
    rank = dist.get_rank()
    print('size: {}  rank: {}'.format(size, rank))
#    init_print(rank, size)

    run(rank, size)
    print('Program End')
