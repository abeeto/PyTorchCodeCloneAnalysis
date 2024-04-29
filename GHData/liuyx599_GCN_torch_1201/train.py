from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.25,
                    help='two-order adj loss penalty coefficient')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, adj2, features, labels, idx_train, idx_val, idx_test = load_data()

acc_results = []
loss_results = []
time_results = []

for i in range(10):
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)


    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        adj2 = adj2.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch, alpha):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        output2 = model(features, adj2)
        loss_train2 = F.nll_loss(output2[idx_train], labels[idx_train])

        loss_train = loss_train + alpha * loss_train2
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


    def test(acc_results, loss_results):
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        acc_results.append("{:.4f}".format(acc_test.item()))
        loss_results.append("{:.4f}".format(loss_test.item()))



    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch, alpha=args.alpha)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    time_results.append("{:.4f}".format(time.time() - t_total))
    # Testing
    test(acc_results, loss_results)


f1 = open("acc_results", "w")
f1.write("\n".join(acc_results))

f2 = open("loss_results", "w")
f2.write("\n".join(loss_results))

f3 = open("time_results", "w")
f3.write("\n".join(time_results))

f1.close()
f2.close()
f3.close()