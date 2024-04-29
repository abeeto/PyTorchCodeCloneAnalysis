import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from sklearn import metrics

from utils import all_metrics


def train(model, args, epoch, data_loader, optimizer):
    print('EPOCH %d' % epoch)
    model.train()
    losses = []

    for batch in data_loader:
        inputs_id, labels, mask = [x for x in batch]
        if args.gpu >= 0:
            inputs_id, labels, mask = inputs_id.cuda(args.gpu), labels.cuda(args.gpu), mask.cuda(args.gpu)

        output, loss = model(inputs_id, labels, mask)

        model.zero_grad()
        # loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def test(model, args, data_loader):
    model.eval()
    losses = []
    y, y_hat = [], []

    for batch in data_loader:
        inputs_id, labels, mask = [x for x in batch]
        if args.gpu >= 0:
            inputs_id, labels, mask = inputs_id.cuda(args.gpu), labels.cuda(args.gpu), mask.cuda(args.gpu)
        output, loss = model(inputs_id, labels, mask)
        # loss = loss_function(output, labels)
        losses.append(loss.item())

        # output = output.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        output = torch.max(output.data, 1)[1].cpu().numpy()
        y.append(labels)
        y_hat.append(output)

    y = np.concatenate(y, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)
    metrics_test = all_metrics(y_hat, y)
    # acc = metrics.accuracy_score(y, y_hat)
    report = metrics.classification_report(y, y_hat, digits=4)
    return metrics_test, report, losses

