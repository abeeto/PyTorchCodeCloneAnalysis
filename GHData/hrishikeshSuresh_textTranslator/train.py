# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
'''
Created on Sat Apr 14 18:34:08 2019

AUTHORS  :   Hrishikesh S.
             C. Anjana Keshav Das
COMMENTS :   # for explanantion
             ## for removing code
             Do not remove code, only comment it
'''
# PATH :: cd "Desktop/Third Year/NLP/Project/ProjectV1"
# To run the code, $ python3 pytorch_train.py
from io import open
import unicodedata
import string
import re
import random
import csv
import sys
import time
import math
import torch
import io
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# use gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to prevent OverflowError: Python int too large to convert to C long
import sys
import csv
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# to indicate start of sentence
SOS_token = 0
# to indicate end of sentence
EOS_token = 1

##csv.field_size_limit(sys.maxsize)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        # Count SOS and EOS
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # to converting words to indices
        # each word in vocabulary will have a 
        # unique index
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# convert a unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(s))
        if unicodedata.category(c) != 'Mn'
    )

# converting all words to lowercase
# trim and remove non-letter characters
def normalizeString(s):
    # convertion to lower case
    s = unicodeToAscii(s.lower().strip())
    # removing punctuations
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    modern = []
    original = []
    # the data is aligned, such that every translation is on the same line
    # extracting the corpus data from data/all_modern.snt.aligned
    with open('data/all_modern.snt.aligned', 'rt', encoding = 'utf-8') as f:
      data = csv.reader(f)
      for row in data:
          if(len(row)==1):
            modern.append(normalizeString(row[0]))
          else:
            string = row[0]
            for i in range(1,len(row)):
              string = string + ',' + row[i]
            modern.append(normalizeString(string))
    print("MODERN READ DONE")
    # extracting the corpus data from data/all_original.snt.aligned
    with open('data/all_original.snt.aligned','rt',  encoding = 'utf-8')as f:
      data = csv.reader(f)
      for row in data:
          if(len(row)==1):
            original.append(normalizeString(row[0]))
          else:
            string = row[0]
            for i in range(1,len(row)):
              string = string + ',' + row[i]
            original.append(normalizeString(string))
    print("ORIGINAL READ DONE")
    ##modern = modern[:3000]
    ##original = original[:3000]

    word_pairs=[]
    # constructing word pairs to show from-to translation
    for i in range(0,len(modern)):
      temp = []
      temp.append(original[i])
      temp.append(modern[i])
      word_pairs.append(temp)
    # to return vocabulary's indices, counts
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
 
    return input_lang,output_lang,word_pairs

MAX_LENGTH = 10

# to remove extremely long strings to prevent 
# training with long sentences as it reduces the
# model's accuracy
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

# helper function
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# prepare data by returning word pairs & indices
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng_shakespeare', 'eng_modern', True)
print(random.choice(pairs))

# seq2seq network/model
# frees us from sequence length and order
# the encoder reads an input sequence and outputs a single vector,
# and the decoder reads that vector to produce an output sequence

# encoder class
# encodes the inputs
# using a GRU unit
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # previous timestep's hidden state
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    # initial hidden state
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# decoder class, using attention mechanism
# uses encoder's output vectors to generate output sequence
# for capturing context
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

# conversion into tensors for training

# preparing list of word indices for each sentence
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# uses indexesFromSentence() to return list of word indices
# for all sentences 
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# uses tensorFromSentence() to return modern & shakespeare
# sentences by splitting them
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# for faster convergence
# can pick up meaning by using just first few words
teacher_forcing_ratio = 0.5

# function to build the model which is used for training
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    # set the gradients to zero before starting to do backpropragation
    # as PyTorch accumulates the gradients on subsequent backward passes
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # using tensor size of input & target to decide
    # input and output for encoder & decoder model respectively
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    # building input layer
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    # teacher forcing is done randomly
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # teacher forcing: feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()  

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# to show minutes while training
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# to show timeSince
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# training the model
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    # start a timer
    start = time.time()
    plot_losses = []
    # Reset every print_every
    print_loss_total = 0
    # Reset every plot_every
    plot_loss_total = 0
    # initialize optimizers and criterion
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # create set of training pairs
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # start empty losses array for plotting
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    # plot losses
    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

# for plotting
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('plots/loss.png')
    plt.show()

# to test the model with custom inputs
# no targets, so we directly expect outputs
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

# to test the model with training corpus data

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# initialize & train
hidden_size = 256
# build encoder
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# build decoder
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# train the models
trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

# attributes of the encoder model
print("Encoder Model's state dict")
for i in encoder1.state_dict():
    print(i, '\t', encoder1.state_dict()[i])

# attributes of the decoder model
print("Attention Decoder Model's state dict")
for i in attn_decoder1.state_dict():
    print(i, '\t', attn_decoder1.state_dict()[i])

# save the models in models folder
torch.save(encoder1.state_dict(), 'models/encoder_pyshake_to_pymodern')
torch.save(attn_decoder1.state_dict(), 'models/attndecoder_pyshake_to_pymodern')

# build encoder
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# build decoder
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# to load the saved load models from models folder
encoder1.load_state_dict(torch.load('models/encoder_pyshake_to_pymodern'))
attn_decoder1.load_state_dict(torch.load('models/attndecoder_pyshake_to_pymodern'))

# for random evaluation
evaluateRandomly(encoder1, attn_decoder1)

# for evaluation along with visualization of attention matrix
output_words, attentions = evaluate(
    encoder1, attn_decoder1, "she is banished")
plt.matshow(attentions.numpy())
plt.savefig('plots/attention.png')

# for visualizing attention matrix
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig("plots/" + input_sentence + ".png")
    ##plt.show()

# evaluate & visualize attention at once - uses previously defined
# showAttention() & evaluate() functions
def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


# sample test 1
evaluateAndShowAttention("what are thou")

# sample test 2
evaluateAndShowAttention("thou shalt die")

# sample test 3
evaluateAndShowAttention("yonder lies the fool")

# sample test 4
evaluateAndShowAttention("what would you have")

