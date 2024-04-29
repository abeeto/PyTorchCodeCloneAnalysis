import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from torch.autograd import Variable


class Parameters:
    # Encapsulation for fundamental Parameters
    def __init__(self, input_lang, output_lang, MAX_LENGTH, EOS_token, UNK_token, SOS_token, USE_CUDA, print_every, plot_every, n_epochs, learning_rate):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = MAX_LENGTH
        self.eos_token = EOS_token
        self.sos_token = SOS_token
        self.unk_token = UNK_token
        self.USE_CUDA = USE_CUDA
        self.print_every = print_every
        self.plot_every = plot_every
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate


def indexes_from_sentence(lang, sentence):
    # Converting sentences to indexes
    return [lang.word2index.get(word, 2) for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence, pars):
    # Converting sentences to indexes
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(pars.eos_token)

    # Preparing indexes variable
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if pars.USE_CUDA:
        var = var.cuda()
    return var


def variables_from_pair(pair, pars):
    # Converting sentence pairs to indexes variables
    input_variable = variable_from_sentence(pars.input_lang, pair[0], pars)
    target_variable = variable_from_sentence(pars.output_lang, pair[1], pars)
    return input_variable, target_variable


def show_plot(points, val_points):
    # Setting up figure
    plt.figure()
    fig, ax = plt.subplots()

    # Putting ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    # Plotting validation steps
    plt.plot(points, 'k-', label='train')
    plt.plot(val_points, 'k--', label='validation')
    plt.legend(loc='best')

    # Printing plot to file
    plt.savefig('plots/lc.png')


def show_attention(input_sentence, output_words, attentions):
    # Setting up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Setting up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Showing label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Printing plot to file
    plt.savefig('plots/attention.png')


def save_model(model, name):
    # Saving a model (but use this just for inference)
    torch.save(model.state_dict(), 'models/'+name+'.pt')


def load_model(model, name):
    # Loading a model (but use this just for inference)
    model.load_state_dict(torch.load('models/'+name+'.pt'))
