
# coding: utf-8

# # Siri Canned Dialog Data Investigation

# In[1]:

#get_ipython().magic(u'matplotlib inline')


# ### 1. Investigation

# In[2]:

def canned_dialogs():
    with open('cannedLog-20170727-20150528.txt', 'r') as f:
        for index, line  in enumerate(f):
            yield index, line


# ** total pairs **

# In[3]:

count  = 0
for index, line in canned_dialogs():
    count = index
print("total", count + 1, "QA pairs")


# In[4]:

utterance_stat = {}
question_stat = {}
answer_stat = {}
for index, line in canned_dialogs():
    part = line.split("\t")
    try:
        question = part[0].strip()
        utterance = part[1].strip()
        answer = part[2].strip()
        if utterance in utterance_stat:
            utterance_stat[utterance] += 1
        else:
            utterance_stat[utterance] = 1
            
        if question in question_stat:
            question_stat[question] += 1
        else:
            question_stat[question] = 1
            
        if answer in answer_stat:
            answer_stat[answer] += 1
        else:
            answer_stat[answer] = 1
    except:
        print("[error]", index, line, part)


# ** utterance stats**

# In[5]:

utterance_stat_sorted = sorted(utterance_stat.items(), key=lambda x: x[1], reverse=True)
for k, v in utterance_stat_sorted:
    print(k, ",", v, ",", str(round(float(v)*100.0 / count, 2)) + "%")


# In[6]:

len(utterance_stat_sorted)


# ** question stats **

# In[7]:

question_stat_sorted = sorted(question_stat.items(), key=lambda x: x[1], reverse=True)
for index, (k, v) in enumerate(question_stat_sorted):
    print(k, ",", v, ",", str(round(float(v)*100.0 / count, 2)) + "%")
    if index == 1000: break


# In[8]:

len(question_stat_sorted)


# ** answer stats **

# In[9]:

answer_stat_sorted = sorted(answer_stat.items(), key=lambda x: x[1], reverse=True)
for index, (k, v) in enumerate(answer_stat_sorted):
    print(k, ",", v, ",", str(round(float(v)*100.0 / count, 2)) + "%")


# In[10]:

len(answer_stat_sorted)


# ### 2. Training data preparation

# In[11]:

filtered_string = "[zh_CN] Couldn't find ducId"
train_question = []
train_utterance = []
train_answer = []
test_couldnt_find = []

for index, line in canned_dialogs():
    try:
        if filtered_string in line or "null" in line:
            test_couldnt_find.append(line)
        else:
            part = line.split("\t")
            question = part[0].strip()
            utterance = part[1].strip()
            answer = part[2].strip()
            train_question.append(question)
            train_utterance.append(utterance)
            train_answer.append(answer)
    except:
        pass


# In[12]:

for i, q in enumerate(train_question):
    print("Q:", q,"\tA:", train_answer[i])
    if i == 19: break


# In[13]:

print("training", len(train_question), str(round(len(train_question)*100.0/count, 2))+"%")
print("testing", len(test_couldnt_find), str(round(len(test_couldnt_find)*100.0/count, 2))+"%")


# ### 3. Experiment

# **prepare data**

# In[14]:

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


# In[15]:

SOS_token = 0
EOS_token = 1

import codecs

def is_alpha(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False

def tokenize(sentence):
    tokens = []
    buf = ""
    for w in sentence:
        if is_alpha(w):
            buf += w
        elif w.strip():
            if len(buf): 
                tokens.append(buf)
                buf = ""
            tokens.append(w)
    if len(buf): tokens.append(buf)
    return tokens

class Siri_Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in tokenize(sentence): # update: zh_CN char as word / u.split(' ')
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
print(" ".join(tokenize(u"若要关闭iPhone，只需按住“睡眠/唤醒”按钮几秒钟，再滑动屏幕上的“滑动来关机”按钮。")))


# In[16]:

def readVocab(siri_question, siri_answer, reverse=False):
    print("Reading Siri pairs...")

    pairs = [(q.decode('utf-8'), siri_answer[i].decode('utf-8')) for i,q in enumerate(siri_question)]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_vocab = Siri_Vocab('siri_answer')
        output_vocab = Siri_Vocab('siri_question')
    else:
        input_vocab = Siri_Vocab('siri_question')
        output_vocab = Siri_Vocab('siri_answer')

    return input_vocab, output_vocab, pairs


# In[17]:

def prepareData(siri_question, siri_answer, reverse=False):
    input_vocab, output_vocab, pairs = readVocab(siri_question, siri_answer, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_vocab.addSentence(pair[0])
        output_vocab.addSentence(pair[1])
    print("Counted words:")
    print(input_vocab.name, input_vocab.n_words)
    print(output_vocab.name, output_vocab.n_words)
    return input_vocab, output_vocab, pairs

input_vocab, output_vocab, pairs = prepareData(train_question, train_answer)


# In[18]:

pair = random.choice(pairs)
print("Q:", pair[0], "A:", pair[1])
pair


# In[19]:

MAX_LENGTH = 0
for p in pairs:
    if len(p[0]) > MAX_LENGTH:
        MAX_LENGTH = len(p[0]) + 2
    if len(p[1]) > MAX_LENGTH:
        MAX_LENGTH = len(p[1]) + 2
print("MAX_LENGTH:", MAX_LENGTH)


# **model**

# In[38]:

from torch.nn import DataParallel
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = DataParallel(self.embedding, device_ids=[0,1,2,3])
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = DataParallel(self.gru, device_ids=[0,1,2,3])

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# In[39]:

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.gru = DataParallel(self.gru, device_ids=[0,1,2,3])
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# **train**

# In[40]:

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in tokenize(sentence)]

def variableFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_vocab, pair[0]) # torch tensor : [2,4,5,1,2,4,6,1,2,6,23, "EOS"]
    target_variable = variableFromSentence(output_vocab, pair[1])
    return (input_variable, target_variable)


# In[41]:

teacher_forcing_ratio = 0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


# In[42]:

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


# In[43]:

def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
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

    showPlot(plot_losses)


# In[44]:

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[45]:

import time
hidden_size = 128
encoder1 = EncoderRNN(input_vocab.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_vocab.n_words,
                               1, dropout_p=0.1)

if use_cuda:
    #print('use cuda')
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()
    #encoder1 = DataParallel(encoder1, device_ids=[0,1,2,3])
    #attn_decoder1 = DataParallel(attn_decoder1, device_ids=[0,1,2,3])
s = time.time()
trainIters(encoder1, attn_decoder1, 1000)
e = time.time()
print(e - s)


# **evaluation**

# In[46]:

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_vocab, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_vocab.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


# In[53]:

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        try:
            # sentence = random.choice(test_couldnt_find)
            # sentence = sentence.decode('utf-8').split("\t")[0]
            sentence = random.choice(pairs)
            print('>', sentence[0])
            output_words, attentions = evaluate(encoder1, attn_decoder1, sentence[0])
            output_sentence = ''.join(output_words)
            print('<', output_sentence)
            print("")
        except:
            print("[error]", sentence, "\n")


# In[59]:

evaluateRandomly(encoder1, attn_decoder1, 10)


# In[63]:

output_words, attentions = evaluate(encoder1, attn_decoder1, "关机")
output_sentence = ''.join(output_words)
print('<', output_sentence)


# In[ ]:



