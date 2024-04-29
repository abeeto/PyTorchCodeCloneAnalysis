"""Utilities for working with Horovod, just to make the coding look better."""
import horovod.torch as hvd
import torch
import torch.nn as nn
import csv
import os
import time
import numpy as np


def printh(*args, **kwargs):
    """Print from the head rank only."""
    if hvd.rank() == 0:
        print(*args, **kwargs)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def save_checkpoint(model, optimizer, epoch, path):
    if hvd.rank() == 0:
        filepath = os.path.join(path,
                                "checkpoint_epoch{epoch}.h5".format(epoch=epoch + 1))
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


def accuracy(output, target, weighted=True, nclass=3):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    if(weighted):
        w_acc = 0.
        nnonzero = 0
        for cl in range(nclass):
            icl = np.where(target.view_as(pred).cpu() == cl)
            if len(icl[0]) != 0:
                w_acc += (pred[icl]).eq(target.view_as(pred)[icl]).cpu().float().sum()/float(len(icl[0]))
                nnonzero += 1
        return w_acc/float(nnonzero)
    else:
        return pred.eq(target.view_as(pred)).cpu().float().mean()


def log_batch(rank, epoch, batch_idx, data_size, local_loss, global_loss,
              local_accuracy, global_accuracy, home_dir, fieldnames,
              total_log_lines):
    filename = os.path.join(home_dir, 'training_rank{:03d}.csv'.format(rank))
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow(dict(time=time.time(), epoch=epoch, rank=rank,
                             batchno=batch_idx,
                             data_size=data_size,
                             local_loss=float("{:.6f}".format(local_loss)),
                             global_loss=float("{:.6f}".format(global_loss)),
                             local_acc=float("{:.6f}".format(local_accuracy)),
                             global_acc=float("{:.6f}".format(global_accuracy))
                             )
                        )
    # only print this to file if there will be less than 100 total lines
    # so twenty five per gpu
    last_epoch_last_batch = ((batch_idx * epoch) == total_log_lines)
    first_epoch_first_batch = ((batch_idx * epoch) == 0)
    batch_epoch_lt_25 = ((batch_idx * epoch) <= 25)
    if not batch_epoch_lt_25:
        batch_epoch_div_by_25 = \
            (batch_idx * epoch) % (total_log_lines // 25) == 0
    else:
        batch_epoch_div_by_25 = False
    if last_epoch_last_batch or first_epoch_first_batch or \
            batch_epoch_lt_25 or batch_epoch_div_by_25:
        logstring = "Epoch {epoch:02d} | Rank {rank:02d} | " \
            .format(epoch=epoch, rank=rank)
        logstring += "batch {batch_idx:03d} | # images {datasize:02d}" \
            .format(batch_idx=batch_idx, datasize=data_size)
        logstring += " | loc_loss {loc_loss:.6f} | glob_loss {loss:.6f}" \
            .format(loc_loss=local_loss, loss=global_loss)
        logstring += " | loc_acc {loc_acc:.6f} | glob_acc {acc:.6f}" \
            .format(loc_acc=local_accuracy, acc=global_accuracy)
        print(logstring)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, train_loader, optimizer, train_set,
                         batches_per_allreduce, warmup_epochs,
                         base_lr):
    if epoch < warmup_epochs:
        epoch += float(batch_idx + 1) / float(train_set.n_batches)
        lr_adj = 1. / hvd.size() \
            * (epoch * (hvd.size() - 1) / warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * hvd.size() * batches_per_allreduce \
            * lr_adj



