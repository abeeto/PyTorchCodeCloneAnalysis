#!/usr/local/anaconda3/envs/experiments/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import sys
import numpy
import datetime
import atexit
import math
from termcolor import colored
import tensorboardX

writer = tensorboardX.SummaryWriter()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
output = 64

if use_cuda:
    print("Using CUDA")
else:
    print("Using CPU")

results = dict()
examined = dict()

keys = []
best = []

search = False
freerun = True


class LSTM(nn.Module):

    def __init__(self, size, hidden, batch, prev, rate):
        super(LSTM, self).__init__()

        self.size = size
        self.rate = rate
        self.batch = 0
        self.epochs = 0
        self.prev = prev

        self.hidden_size = hidden * batch

        self.forget_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.forget_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.input_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.input_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.state_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.state_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.output_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.output_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.tanh = torch.nn.Tanh()

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)

        self.steps = 0
        self.generations = 0

        self.hidden = torch.zeros(self.hidden_size, device=device)

    def reset(self):
        self.context = torch.zeros(self.hidden_size, device=device)

    def forward(self, x):
        # detach state tensors
        self.context = self.context.detach()
        self.hidden = self.hidden.detach()

        # layers
        x = x.detach().view(-1).cuda()

        # process layers
        f = torch.add(self.forget_input(x), self.forget_hidden(self.hidden))
        i = torch.add(self.input_input(x), self.input_hidden(self.hidden))
        s = torch.add(self.state_input(x), self.state_hidden(self.hidden))
        o = torch.add(self.output_input(x), self.output_hidden(self.hidden))

        # activations
        f = self.sigmoid(f)
        i = self.sigmoid(i)
        s = self.tanh(s)
        o = self.sigmoid(o)

        # gating mechanism
        self.context = (f * self.context) + (i * s)

        # tanh output
        self.hidden = self.tanh(self.context) * o

        return self.hidden.clone()
class Model(nn.Module):

    def __init__(self, size, prev, batch_size, dropout, rate, hidden):
        super(Model, self).__init__()

        self.rnn1 = LSTM(size, hidden, batch_size, prev, rate).cuda()

        self.d = dropout
        self.r = rate

        self.epochs = 1
        self.batches = 1
        self.counter = 0
        self.runs = 0
        self.count = 0

        self.rate = rate

        self.output_decoder = torch.nn.Linear(hidden * batch_size, size * batch_size).cuda()
        self.dropout = torch.nn.Dropout(dropout)

        self.field = [text[x] for x in range(n_prev)]

        self.clear_internal_states()

        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam([
        #    {'params': self.parameters()},
        #    ], weight_decay=0.0, lr=rate)
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=rate, momentum=momentum)
        self.initialize_weights()

        # std = 1.0/math.sqrt(self.rnn.hidden_size)

        # for p in self.parameters():
        #    p.data.uniform_(-std, std)

    def clear_internal_states(self):
        self.rnn1.reset()

    def initialize_weights(self):
        self.output_decoder.bias.data.fill_(0)
        self.output_decoder.weight.data.uniform_(-1, 1)

    def get_input_vector(self, chars):
        out = []

        for c in chars:
            out.append(one_hot(c))

        out = torch.stack(out).cuda()
        return out

    def forward(self, inp):
        x = torch.autograd.Variable((inp).view(-1))

        x = self.rnn1(x)
        x = self.dropout(x)

        x = self.output_decoder(x)
        x = x.view(nbatches, -1)

        return x


def splash(a):
    if a:
        print(
            "RNN Text Generator\nUsage:\n\n-f --filename: filename of input text - required\n-h --hidden: number of hidden layers, default 1\n-r --rate: learning rate\n-p --prev: number of previous states to observe, default 0.05")
        print("\nExample usage: <command> -f input.txt -h 5 -r 0.025")
    else:
        print("\nRNN Text Generator\n")
        print("Alphabet size: {}".format(alphabet_size))

        print("Hyperparameters:")
        params = sys.argv
        params.pop(0)

        for a in list(params):
            print(a, " ", end="")

        print("\n")
        print(datetime.datetime.now())
        print("\n")

def getIndexFromLetter(letter, list):
    return list.index(letter)

def getLetterFromIndex(i, list):
    return list[i]

def parse(args, arg):
    for i in range(len(args)):
        if args[i] in arg:
            if len(args) < i + 1:
                return ""
            if args[i + 1].startswith("-"):
                splash(True)
            else:
                return args[i + 1]

    return False

def savemodel():
    print("Save model parameters? [y/n]➡")
    filename_input = input()

    if filename_input == 'y' or filename_input == 'Y' or filename_input.lower() == 'yes':
        filename = "Model-" + str(datetime.datetime.now()).replace(" ", "_")
        print("Save as filename [default: {}]➡".format(filename))

        filename_input = input()
        if not filename_input == "":
            filename = "Model-" + str(filename_input).replace(" ", "_")

        print("Saving model as {}...".format(filename))
        modelname = "./models/{}".format(filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()
        }, modelname)

    # print("Best parameters:")
    # print(best)
    quit()

def loadmodel():
    print("Load")
    # load model parameters if checkpoint specified
    if not model_filename == False:
        try:
            checkpoint = torch.load(model_filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except FileNotFoundError:
            print("Model not found.")
            quit()
    else:
        print("New model")

atexit.register(savemodel)
model_filename = None

try:
    model_filename = parse(sys.argv, ["--load", "-l"])
    filename = parse(sys.argv, ["--filename", "-f"])
    if not filename or filename == "":
        splash()
    rate = float(parse(sys.argv, ["--rate", "-r"]))
    if not rate or rate == "":
        rate = 0.012345678912121212
    hidden = int(parse(sys.argv, ["--hidden", "-h"]))
    if not hidden or hidden == "":
        hidden = 512
    nbatches = int(parse(sys.argv, ["--batch", "-b"]))
    if not nbatches:
        nbatches = 2
    momentum = float(parse(sys.argv, ["--momentum", "-m"]))
    if not momentum:
        momentum = 0.4
    n_prev = int(parse(sys.argv, ["--previous", "-p"]))
    if not n_prev:
        n_prev = 9
    dropout = float(parse(sys.argv, ["--dropout", "-d"]))
    if not dropout:
        dropout = 0.5
    temperature = float(parse(sys.argv, ["--temp", "-t"]))
    if not temperature:
        temperature = 0.96

except:
    splash(True)
    quit()

alphabet = [' ', '!', '"', '#', '$', '%', '&', "'",
            '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', ':', ';', '<', '=', '>', '?',
            '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z', '[', ']', '^', '_', 'a',
            'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', '|', '~', '¶']
text = []
e = 0
c = 0

# open file
with open(filename, "r") as f:
    # reads all lines and removes non alphabet words
    intext = f.read()

for l in list(intext):
    if l == "\n": l = "¶"
    if l == "\x1b": print("XXX")
    text.append(l)

for l in text:
    sys.stdout.flush()

    if l not in alphabet:
        alphabet.append(l)
        print("\r{}% - {}/{}".format(int(c / len(text) * 100), c, len(text)), end="")
    c += 1

alphabet.sort()

alphabet_size = len(alphabet)
model = Model(alphabet_size, n_prev, nbatches, dropout, rate, hidden).cuda()
if not model_filename == None:
    loadmodel()

# encode vector from char
def one_hot(char):
    output = torch.zeros(alphabet_size).cuda()
    output[alphabet.index(char)] = 1

    return output

# get output char from vectors
def get_output(inp, t):
    inp = torch.nn.Softmax(dim=0)((inp / t).exp())
    sample = torch.multinomial(inp / inp.sum(), 1)[:]

    return alphabet[sample]

def generate(model, temperature):

    model.steps = 1

    while True:
        inp = model.get_input_vector(model.field)
        inp = [inp for _ in range(nbatches)]
        inp = torch.stack(inp)
        out = model.forward(inp)

        # get outputs
        char = []

        for o in out.split(1):
            a = get_output(o[0], temperature)
            char.append(a)

        c = char[0]

        print(c, end="")
        sys.stdout.flush()

        if model.steps > 40:
            model.field.append(c)
            model.field.pop(0)
        else:
            new_letter = alphabet.index(text[(model.steps + 1) % len(text)])
            model.field.append(alphabet[new_letter])
            model.field.pop(0)

        model.steps += 1

generate(model, temperature)