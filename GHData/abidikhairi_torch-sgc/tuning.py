import torch as th
import torchmetrics.functional as thm
import torch.nn.functional as F
from hyperopt import hp, fmin, tpe, STATUS_OK
from math import log
from models import SGC


def train(model, n_feats, labels, train_idx, test_idx, weight_decay):
    optimizer = th.optim.Adam(model.parameters(), lr=0.2, weight_decay=weight_decay)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(n_feats)
        logits = F.log_softmax(output, dim=1)
        trainloss = F.nll_loss(logits[train_idx], labels[train_idx])

        trainloss.backward()
        optimizer.step()
    
    model.eval()
    with th.no_grad():
        output = model(n_feats)
        logits = F.log_softmax(output, dim=1)
        
        testloss = F.nll_loss(logits[test_idx], labels[test_idx])
        testacc = thm.accuracy(logits[test_idx], labels[test_idx])
    
    return testloss, testacc


def sgc_objective(space):
    model =  SGC(space['in_feats'], space['out_feats'], space['adj'], space['k'], space['bias'])
    testloss, testacc = train(model, space['n_feats'], space['labels'], space['train_idx'], space['test_idx'], space['weight_decay'])

    print("Test loss: {}, Test accuracy: {} Weight decay: {}".format(testloss, testacc, space['weight_decay']))

    return {'loss': testloss, 'acc': testacc, 'status': STATUS_OK}


def main():
    adj = th.load('data/cora/adj.pt')
    labels = th.load('data/cora/labels.pt')
    n_feats = th.load('data/cora/feats.pt')
    train_idx = th.load('data/cora/train_idx.pt')
    test_idx = th.load('data/cora/test_idx.pt')

    space = {}
    space['in_feats'] = n_feats.shape[1]
    space['out_feats'] = labels.max().item() + 1
    space['adj'] = adj
    space['k'] = hp.choice('k', [1, 2, 3, 4])
    space['bias'] = True
    space['n_feats'] = n_feats
    space['labels'] = labels
    space['train_idx'] = train_idx
    space['test_idx'] = test_idx
    space['weight_decay'] = hp.loguniform('weight_decay', log(1e-10), log(1e-5))

    best = fmin(sgc_objective, space=space, algo=tpe.suggest, max_evals=60)
    print("Best weight decay: {:.2e}".format(best["weight_decay"]))

if __name__ == '__main__':
    main()