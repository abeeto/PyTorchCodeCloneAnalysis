import os
import shutil

import torch
import torch.nn.parallel
import torch.utils.data


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


# save the checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', path='.'):
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)
    print ("saving checkpoint to {}".format(filepath))
    if is_best:
        print ("New Best!")
        shutil.copyfile(filepath, os.path.join(path,'model_best.pth.tar'))
