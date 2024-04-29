from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

device = torch.device("cpu")

MAX_LENGTH = 100

SOS_token = 0
EOS_token = 1

class Para:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "#", 1: "$", 2: " "}
        self.n_chars = 3
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            for character in word:
                self.addChar(character)
    def addChar(self, char1):
        if char1 not in self.char2index:
            self.char2index[char1] = self.n_chars
            self.char2count[char1] = 1
            self.index2char[self.n_chars] = char1
            self.n_chars += 1
        else:
            self.char2count[char1] += 1

def normalizeString(s):
    korean = re.compile('[^가-힣^a-z^A-Z^0-9^.,!?/:~()^"\"^" "]')
    result = korean.sub('', s)
    return result

def readText():
    inputs = open('data/ques.txt', encoding = 'utf-8').read().strip().split('\n')
    outputs = open('data/ans.txt', encoding = 'utf-8').read().strip().split('\n')
    inputs = [normalizeString(s) for s in inputs]
    outputs = [normalizeString(s) for s in outputs]
    print(inputs, outputs)

    inp = Para('input')
    outp = Para('output')
    pair = []
    for i in range(len(inputs)):
        pair.append([inputs[i], outputs[i]])
    print(pair)
    return inp, outp, pair

def prepareData():
    input_para, output_para, pairs = readText()
    for pair in pairs:
        input_para.addSentence(pair[0])
        output_para.addSentence(pair[1])
    print("Counted chars : ")
    print(input_para.name, input_para.n_chars)
    print(output_para.name, output_para.n_chars)
    return input_para, output_para, pairs

input_para, output_para, pairs = prepareData()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(para, sentence):
    arr = []
    for chars in sentence.split(' '):
        for char1 in chars:
            arr.append(para.char2index[char1])
    return arr
    #return [sent.char2index[char1] for char1 in sentence.split(' ')]

def tensorFromSentence(para, sentence):
    indexes = indexesFromSentence(para, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_para, pair[0])
    target_tensor = tensorFromSentence(output_para, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    print(learning_rate)
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    #training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = []
    for j in range(int(n_iters / len(pairs)) + 1):
        for i in pairs:
            training_pairs.append(tensorsFromPair(i))

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_tensor = tensorFromSentence(input_para, sentence)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi == EOS_token:
            #decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_para.index2char[topi.item()])
        decoder_input = topi.squeeze().detach()

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    count = 0
    for pair in pairs:
        print('>', pair[0])
        print('=', pair[1])
        output_chars, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_chars)
        print('<', output_sentence)
        if pair[1] == output_sentence:
            count += 1
    print(count / len(pairs) * 100, "% accuracy")

def inputsentence(encoder, decoder, sentence):
    output_chars, attentions = evaluate(encoder, decoder, sentence)
    output_sentence = ''.join(output_chars)
    print('<', output_sentence)
    return output_sentence

hidden_size = 100
encoder1 = EncoderRNN(input_para.n_chars, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_para.n_chars, dropout_p=0.1).to(device)
trainIters(encoder1, attn_decoder1, 3000, print_every=100, learning_rate = 0.01)
print("Training complete")

def evaluateSeperately(encoder1, attn_decoder1):
    while True:
        sentence = input()
        sentence = normalizeSentence2(sentence)
        out1 = inputsentence(encoder1, attn_decoder1, sentence)
evaluateRandomly(encoder1, attn_decoder1, 100)