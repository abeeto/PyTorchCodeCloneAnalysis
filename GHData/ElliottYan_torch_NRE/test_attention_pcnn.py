'''
It actually is the CNN + Attention model
We can use this to easily implement CNN, CNN + att + T
'''

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.metrics as metrics

from dataset import Dataset
from dataset import collate_fn
import numpy as np

test_data = Dataset(root, train_test='test')
test_loader = data.DataLoader(datasets, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
model.eval()

prec = []
outs = []
y_true = []

for batch_ix, (bags, labels) in enumerate(test_loader):
    # if batch_ix > 10:
    #     break
    out = model(bags, datasets.max_sent_len, labels=labels)
    pdb.set_trace()
    outs.append(out.cpu().data.numpy())
    y_true.append(labels.numpy())
    pred = out.max(dim=1)[1].long().data
    bz = pred.size()[0]
    correct = pred.eq(labels.cuda())
    acc = float(correct.sum()) / bz
    prec.append(acc)
    print("Accuracy in %d batch:%f"%(batch_ix, acc))

prec = sum(prec) / len(prec)
print("Average test accuracy is %f"%prec)

# draw precision-recall curve

y_pred = np.concatenate(outs, axis=0)
y_true = np.concatenate(y_true, axis=0)

np.save('./result/labels_max_len.npy', y_true)
np.save("./result/prediction_max_len.npy", y_pred)
