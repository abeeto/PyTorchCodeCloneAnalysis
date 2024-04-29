import os
import time
import argparse
import torch as th
import torch.nn.functional as F
from models import GCN, SGC
from utils import test, train


def main(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    adj = th.load(os.path.join(args.data, "adj.pt")).to(device)
    feats = th.load(os.path.join(args.data, "feats.pt")).to(device)
    labels = th.load(os.path.join(args.data, "labels.pt")).to(device)
    train_idx = th.load(os.path.join(args.data, "train_idx.pt")).to(device)
    test_idx = th.load(os.path.join(args.data, "test_idx.pt")).to(device)

    nfeats = feats.shape[1]
    nhids = args.hidden
    nclasses = labels.max().item() + 1

    sgc = SGC(nfeats, nclasses, adj, 2).to(device)
    optimizer = th.optim.Adam(sgc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss

    sgc_training_time = []
    gcn_training_time = []

    for epoch in range(args.epochs):
        tick = time.time()
        train(sgc, optimizer, loss_fn, feats, labels, train_idx)
        tock = time.time()
        sgc_training_time.append(tock - tick)

    gcn = GCN(nfeats, nhids, nclasses, adj).to(device)
    optimizer = th.optim.Adam(gcn.parameters(), lr=0.1, weight_decay=5e-4)
    loss_fn = F.nll_loss

    for epoch in range(args.epochs):
        tick = time.time()
        train(gcn, optimizer, loss_fn, feats, labels, train_idx)
        tock = time.time()
        gcn_training_time.append(tock - tick)


    print("SGC training time: {} ms".format(th.tensor(sgc_training_time).mean().item()))
    print("GCN training time: {} ms".format(th.tensor(gcn_training_time).mean().item()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data/cora', help='path to dataset') 
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--hidden', type=int, default=16, help='number of hidden units')

    args = parser.parse_args()
    main(args)
