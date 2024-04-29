from os import path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_batch(data, batch_size):
    nbatch = len(data) // batch_size
    trim_num = nbatch * batch_size
    data = np.asarray(data[:trim_num]).reshape([batch_size, nbatch])
    return data


def get_batch(opt, data, i):
    # data: [batch_size, original_seq_len]
    # We need to get smaller data seq according to bptt_len
    seq_len = min(opt.bptt_len, data.shape[1] - i - 1)
    context = data[:, i:i+seq_len]
    target = data[:, i+1:i+1+seq_len].reshape(-1)
    return context, target


def plot_loss_fig(opt, train_loss, eval_loss):
    train_x = [(i + 1) * opt.plot_span for i in range(len(train_loss))]
    eval_x = [(i + 1) * opt.eval_span for i in range(len(eval_loss))]
    plt.plot(train_x, train_loss, label='train_loss')
    plt.plot(eval_x, eval_loss, label='eval_loss')
    plt.legend()
    plt.savefig(path.join(opt.save_dir, 'loss.png'))
