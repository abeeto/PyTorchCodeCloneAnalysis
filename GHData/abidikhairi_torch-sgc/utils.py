import torch as th
import torch.nn.functional as F


def accuracy(pred, labels):
    return th.sum(pred.argmax(dim=1) == labels) / len(pred)


def train(model, optimizer, loss_fn, feats, labels, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(feats)
    logits = F.log_softmax(out, dim=1)
    
    loss = loss_fn(logits[train_idx], labels[train_idx])
        
    loss.backward()
    optimizer.step()


def test(model, loss_fn, feats, labels, test_idx):
    with th.no_grad():
        model.eval()
        out = model(feats)
        logits = F.log_softmax(out, dim=1)

        test_loss = loss_fn(logits[test_idx], labels[test_idx]).item()
        acc = accuracy(logits[test_idx], labels[test_idx]).item()

    return test_loss, acc
