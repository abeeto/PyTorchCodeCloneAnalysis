# MIT License
# 
# Copyright (c) 2017 Max W. Y. Lam
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import importlib
import numpy as np

import torch
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d
from torch.autograd import Variable
from torch.utils.data import DataLoader

from contrib import SGPA, TrainSampler, ValidationSampler, TestSampler


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--hidden-layers', type=list, default=[100, 100], nargs='+',
    metavar='L', help='depth and widths of hidden layers (default: [100, 50])')
parser.add_argument('--n-basis', type=int, default=50, metavar='N',
    help='number of spectral basis in SGPA (default: 50)')
parser.add_argument('--train-samples', type=int, default=12, metavar='N',
    help='number of samples in training SGPA (default: 12)')
parser.add_argument('--test-samples', type=int, default=256, metavar='N',
    help='number of samples in testing SGPA (default: 256)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
    help='number of epochs to train (default: 5000)')
parser.add_argument('--tolerance', type=int, default=20, metavar='N',
    help='tolerance parameter for early stopping (default: 10)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
    help='disables CUDA training or not (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    help='number of epochs to wait per log (default: 10)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

task_name = input('Enter Task Name: ')
module = importlib.import_module(task_name)
dataset = module.Dataset()

train_sampler = TrainSampler(dataset)
train_loader = DataLoader(dataset, args.batch_size, sampler=train_sampler)
valid_sampler = ValidationSampler(train_sampler)
valid_loader = DataLoader(dataset, len(valid_sampler), sampler=valid_sampler)
test_sampler = TestSampler(valid_sampler)
test_loader = DataLoader(dataset, len(test_sampler), sampler=test_sampler)

if args.cuda:
    model.cuda()

D_in, D_out = dataset.data_tensor.size(1), dataset.target_tensor.size(1)
layer_sizes = [D_in]+args.hidden_layers+[D_out]

model = torch.nn.Sequential()
for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    if(l < len(layer_sizes)-2):
        model.add_module('linear_l'+str(l), Linear(n_in, args.n_basis))    
        model.add_module('sgpa_l'+str(l), SGPA(args.n_basis, 100, False))
        # model.add_module('linear_l'+str(l), Linear(n_in, n_out))
        # model.add_module('dropout_l'+str(l), Dropout(0.2))
        # model.add_module('relu_l'+str(l), ReLU())
        continue
    
    model.add_module('linear_l'+str(l), Linear(n_in, n_out))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_dict = {'min_val_loss':np.Infinity, 'no_improve':0, 'best_state':None}
loss_fn = torch.nn.MSELoss(size_average=True)

def train(epoch, train_dict):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        for _ in range(args.train_samples-1):
            output += model(data)
        minibatch_train_loss = loss_fn(output/args.train_samples, target)
        minibatch_train_loss.backward()
        optimizer.step()
        train_loss += minibatch_train_loss
    train_loss /= len(train_loader)
    if epoch % args.log_interval == 0:
        valid_loss = 0
        for batch_idx, (data, target) in enumerate(valid_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            for _ in range(args.test_samples-1):
                output += model(data)
            valid_loss += loss_fn(output/args.test_samples, target)
        valid_loss /= len(valid_loader)
        if(valid_loss.data[0] < train_dict['min_val_loss']):
            train_dict['no_improve'] = 0
            train_dict['min_val_loss'] = valid_loss.data[0]
            train_dict['best_state'] = model.state_dict()
        else:
            train_dict['no_improve'] += 1
        print('Train Epoch: {:05d} Train Loss: {:.5f} Valid Loss: {:.5f}'
            ' Early Stop: {}/{}'.format(
                epoch, train_loss.data[0], valid_loss.data[0],
                    train_dict['no_improve'], args.tolerance))

def test():
    # model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        for _ in range(args.test_samples-1):
            output += model(data)
        output = output.data*dataset.target_std/args.test_samples
        target = target.data*dataset.target_std
        test_loss += torch.sum((output-target)**2)/len(target)
    test_loss /= len(test_loader)
    print('\nTest set: Average Loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch, train_dict)
    # test()
    if(train_dict['no_improve'] >= args.tolerance):
        break
model.load_state_dict(train_dict['best_state'])
test()