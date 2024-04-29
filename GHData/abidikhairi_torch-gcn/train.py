import os
import argparse
import torch as th
import torch.nn.functional as F
from models import GCN
from utils import accuracy


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

    model = GCN(nfeats, nhids, nclasses, adj).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss


    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(feats)
        logits = F.log_softmax(out, dim=1)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        
        loss.backward()
        optimizer.step()

    with th.no_grad():
        model.eval()
        out = model(feats)
        logits = F.log_softmax(out, dim=1)

        test_loss = loss_fn(logits[test_idx], labels[test_idx]).item()
        acc = accuracy(logits[test_idx], labels[test_idx]).item()

        print("Test: Loss: {:.4f}, Acc: {:.4f} %".format(test_loss, acc*100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data/cora', help='path to dataset') 
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--hidden', type=int, default=16, help='number of hidden units')

    args = parser.parse_args()
    main(args)
