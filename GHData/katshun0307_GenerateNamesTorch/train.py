# -*- coding: utf-8 -*- #

"""
"""

import os
import time
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import dataset
from network import RNN

dataset = dataset.Dataset()

rnn = RNN(dataset.n_letters, 128, dataset.n_letters, dataset.n_categories)

criterion = nn.NLLLoss()

learning_rate = 0.0005

rnn.train()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def savepoint(iter, total_loss):
    torch.save({'epoch': iter,
                'nn_state_dict': rnn.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'loss': total_loss
                }, "models/gen_names.pkl")


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


n_iters = 10000
initial_iter = 1
print_every = 1000
plot_every = 1000
save_every = 1000
all_losses = []
total_loss = 0  # Reset every plot_every iters

# load past data
if os.path.exists("models/gen_names.pkl"):
    checkpoint = torch.load("models/gen_names.pkl")
    rnn.load_state_dict(checkpoint['nn_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    initial_iter = checkpoint['epoch']
    total_loss = checkpoint['loss']
    print("checkpoint loaded")

start = time.time()

for iter in range(initial_iter, initial_iter + n_iters + 1):
    output, loss = train(*dataset.random_training_example())
    total_loss += loss
    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, (iter - initial_iter) / n_iters * 100, loss))
    if iter % plot_every == 0:
        all_losses.append((iter, total_loss / plot_every))
        total_loss = 0
    if iter % save_every == 0:
        savepoint(iter, all_losses)

savepoint(initial_iter + n_iters, all_losses)

plt.figure()
plt.plot(all_losses)

plt.show()