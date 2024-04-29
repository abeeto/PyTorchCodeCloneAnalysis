import torch
import torch.nn as nn
import unicodedata
import string
from io import open
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob


path = "data/names/"
ext = "*.txt"


def get_files(path):
    return glob.glob(os.path.join(path, ext))


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodetoascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)


# Build the category_lines dictionary, a list of names per language
category_names = {}
all_categories = []


# Read a file and split into lines
def readfiles(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodetoascii(line) for line in lines]


for f in get_files(path):
    category = f.split('/')[2].split('.')[0]
    all_categories.append(category)
    lines = readfiles(f)
    category_names[category] = lines

n_categories = len(all_categories)

# converting names to tensors


def lettertoindex(l):
    """converts letter to index"""
    return all_letters.index(l)


def lettertotensor(l):
    """converts a letter to a tensor"""
    tensor = torch.zeros(1, len(all_letters))
    l_idx = lettertoindex(l)
    tensor[0][l_idx] = 1
    return tensor


def nametotensor(name):
    """converts a name into a tensor of shape seq, 1, len(all_letters)"""
    tensor = torch.zeros(len(name), 1, len(all_letters))
    for idx, l in enumerate(name):
        tensor[idx][0][lettertoindex(l)] =1
    return tensor


def categoryfromoutput(output):
    top_v, top_i = output.topk(1)
    cat_i = top_i[0].item()
    return all_categories[cat_i], cat_i


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def inithidden(self):
        return torch.zeros(1, self.hidden_size)


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):

        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers =num_layers

        self.gru  = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.size(1)

        hidden = self.inithidden(batch_size)
        output, hidden = self.gru(x, hidden)
        fc_out = self.fc(hidden)
        return fc_out

    def inithidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Instantiate models here: from file import


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
gru = GRU(n_letters, n_hidden, n_categories)


def random_choice(lst):
    return lst[np.random.randint(0, len(lst)-1)]


def randomtrainningexample():
    category = random_choice(all_categories)
    name = random_choice(category_names[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    name_tensor = nametotensor(name)
    return category, name, category_tensor, name_tensor


# setting hyperparameters
learning_rate = 0.005
criterion_rnn = nn.NLLLoss()
criterion_gru = nn.CrossEntropyLoss()


def train_rnn(category_tensor, name_tensor):
    hidden = rnn.inithidden()

    rnn.zero_grad()

    for i in range(name_tensor.size()[0]):
        output, hidden = rnn.forward(name_tensor[i], hidden)

    loss = criterion_rnn(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)  # can also use torch.optim() if you so choose to

    return output, loss.item()


def train_gru(category_tensor, name_tensor):

    gru.zero_grad()

    output = gru.forward(name_tensor)

    loss = criterion_gru(output.squeeze(1), category_tensor)
    loss.backward()

    for p in gru.parameters():
        p.data.add_(-learning_rate, p.grad.data)  # can also use torch.optim() if you so choose to

    return output, loss.item()


def time_taken(start):
    time_elapsed = time.time() - start
    min = time_elapsed//60
    sec = time_elapsed%60
    return '%dm %ds' % (min, sec)


def evaluate(name_tensor, model):

    if model == rnn:

        hidden = rnn.inithidden()

        for i in range(name_tensor.size()[0]):
            output, next_hidden = rnn.forward(name_tensor[i], hidden)

    elif model == gru:

        output = gru.forward(name_tensor)

    return output


def predict(name, model, n_predictions=3):
    print(name)

    with torch.no_grad():
        output = evaluate(nametotensor(name), model)

        if model == gru:
            output = output.squeeze(1)

        top_n, top_i = output.topk(n_predictions, 1, True)
        predictions_lst = []

        for i in range(n_predictions):
            val = torch.exp(top_n[0][i]) if model == rnn else top_n[0][i]
            cat_idx = top_i[0][i].item()
            print(f'Value: {val.item()}, language: {all_categories[cat_idx]}')
            predictions_lst.append([val, all_categories[cat_idx]])


def main():
    for i in range(10):
        category, name, category_tensor, name_tensor = randomtrainningexample()
        print(f'category: {category}, name: {name}')

    print(f'RNN: {rnn}')
    print(f'GRU: {gru}')

    name = nametotensor('Albert')
    hidden = torch.zeros(1, n_hidden)

    for i in range(name.size()[0]):
        output, next_hidden = rnn.forward(name[i], hidden)

    print(f'output: {output}, next_hidden: {next_hidden}')

    output = gru.forward(name)
    print(output)

    # Training
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    current_loss_rnn = 0
    all_losses_rnn = []

    start = time.time()

    print('Training Vanilla RNN')
    print(' ')

    # Training Vanilla RNN
    for i in range(1, n_iters+1):
        category, name, category_tensor, name_tensor = randomtrainningexample()
        output, loss = train_rnn(category_tensor, name_tensor)
        current_loss_rnn += loss

        if n_iters % print_every == 0:
            pred, pred_i = categoryfromoutput(output)
            prediction = 'True' if pred == category else f'False, correct one is {category}'
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, time_taken(start), loss, name, pred, prediction))

        if i % plot_every == 0:
            all_losses_rnn.append(current_loss_rnn/plot_every)
            current_loss_rnn = 0

    print('')
    print('############################################################################')
    print('Training GRU network now')

    # Training Gated Recurrent Unit
    current_loss_gru = 0
    all_losses_gru = []

    for i in range(1, n_iters+1):
        category, name, category_tensor, name_tensor = randomtrainningexample()
        output, loss = train_gru(category_tensor, name_tensor)
        current_loss_gru += loss

        if n_iters % print_every == 0:
            pred, pred_i = categoryfromoutput(output)
            prediction = 'True' if pred == category else f'False, correct one is {category}'
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, time_taken(start), loss, name, pred, prediction))

        if i % plot_every == 0:
            all_losses_gru.append(current_loss_gru/plot_every)
            current_loss_gru = 0

    # plot confusion matrix and losses
    confusion_rnn = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Add one to each row: the real category. Each column: the predicted category.
    # The darker the principal diagonal, the better the model.
    for i in range(n_confusion):
        category, name, category_tensor, name_tensor = randomtrainningexample()
        output_rnn = evaluate(name_tensor, rnn)
        guess, guess_i_rnn = categoryfromoutput(output_rnn)
        real_category_i = all_categories.index(category)
        confusion_rnn[real_category_i][guess_i_rnn] += 1

    for i in range(n_categories):
        confusion_rnn[i] = confusion_rnn[i]/confusion_rnn[i].sum()

    # Set up fig, axes.
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    cax = ax1.matshow(confusion_rnn.numpy())
    fig.colorbar(cax)

    # Set the labels for x and y axes
    ax1.set_xticklabels([''] + all_categories, rotation=90)
    ax1.set_yticklabels([''] + all_categories)

    # Major tick locations on the axis are set.
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Plot Vanilla Rnn losses
    ax1 = fig.add_subplot(222)
    ax1.set_title('Vanilla Rnn Losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Losses')
    ax1.plot(all_losses_rnn)

    # Gated Recurrent unit
    confusion_gru = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Add one to each row: the real category and each column: the predicted category.
    # The darker the principal diagonal, the better the model.
    for i in range(n_confusion):
        category, name, category_tensor, name_tensor = randomtrainningexample()
        output_gru = evaluate(name_tensor, gru)
        guess, guess_i_gru = categoryfromoutput(output_gru)
        real_category_i = all_categories.index(category)
        confusion_gru[real_category_i][guess_i_gru] += 1

    for i in range(n_categories):
        confusion_gru[i] = confusion_gru[i]/confusion_gru[i].sum()

    ax1 = fig.add_subplot(223)
    cax1 = ax1.matshow(confusion_gru.numpy())
    fig.colorbar(cax1)

    # Set the labels for x and y axes
    ax1.set_xticklabels([''] + all_categories, rotation=90)
    ax1.set_yticklabels([''] + all_categories)

    # Major tick locations on the axis are set.
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Plot GRU losses
    ax1 = fig.add_subplot(224)
    ax1.set_title('GRU losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Losses')
    ax1.plot(all_losses_gru)

    plt.show()

    # predict for vanilla Rnn
    predict('Akutagawa', rnn)
    predict('Avgerinos', rnn)
    predict('Lestrange', rnn)

    # predict for GRU
    predict('Akutagawa', gru)
    predict('Avgerinos', gru)
    predict('Lestrange', gru)


if __name__ == '__main__':
    main()
