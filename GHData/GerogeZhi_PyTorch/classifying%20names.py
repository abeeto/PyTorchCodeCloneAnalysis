
# -----------preparing the data

from __future__ import unicode_literals,print_function,division
from io import open
import glob

def findFiles(path): return glob.glob(path)
print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + '.,;'
n_letters = len(all_letters)

# turn a unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)!='Mn'
        and c in all_letters
    )
print(unicodeToAscii('Ślusàrski'))

# build the category_lines dictionary, a list of names per language

category_lines={}
all_catagories=[]

# read a file and split into lines

def readlines(filename):
    lines = open(filename,encoding = 'utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)

print(category_lines['Italian'][:5])

#-----------Turning Names into Tensors

import torch
# find letter index from all_letters
def letterToIndex(letter):
    return all_letters.find(letter)
# just for demostration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)]=1
    return tensor
# turn a line into a <line_length X 1 X n_letters>
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line),1,n_letters)
    for li,letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)]=1
    return tensor
print(letterToTensor('J'))
print(lineToTensor('Jones').size())

# -----------------creating the network

import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size,output_size)
        self.softmax = nn.LogSoftMax(dim=1)
    def forward(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output,hidden
    def initHidden(self):
        return Variable(torch.zeros(1,self.hidden_size))
n_hidden = 128
rnn = RNN(n_letters , n_hidden, n_categories)

input = Variable(letterToTensor('A'))
hidden = Variable(torch.zeros(1,n_hidden))
output , next_hidden = rnn(input, hidden)

#For the sake of efficiency we don’t want to be creating a new Tensor for every step,

input = Variable(lineToTensor('Albert'))
hidden = Variable(torch.zeros(1,n_hidden))
output,next_hidden = rnn(input[0],hidden)
print(output)
#where every item is the likelihood of that category (higher is more likely).

# ---------------training

# preparing for training
def categoryFromOutput(output):
    # use Tensor.topk to get the index of the greatest value:
    top_n,top_i = output.data.topk(1) # Tensor our of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i],category_i

print(categoryFromOutput(output))

# a quick way to get a training example (a name and its language)

import random

def randomChoice(l):
    return l[random.randint(0,len(1)-1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))    
    line_tensor = Variable(lineToTensor(line))
    return category,line,category_tensor,line_tensor

for i in range(10):
    category,line,category_tensor,line_tensor = randomTrainingExample()
    print('category = ',category,'/line=',line)

# training the network

criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor,line_tensor):
    hidden=rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output.hidden = rnn(line_tensor[i],hidden)
    loss = criterion(output,category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-learning_rate,p.grad.data)
    return output,loss.data[0]

import time
import math

n_iters = 10000
print_every = 5000
plot_every = 1000

# keep track of losses for plotting
current_loss = 0
all_losses=[]

def timesince(since):
    now = time.time()
    s = now-since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds'%(m,s)

start = time.time()

for iter in range(1,n_iters+1):
    category,line,category_tensor,line_tensor = randomTrainingExample()
    output,loss = train(category_tensor,line_tensor)
    current_loss += loss
    
    if iter % print_every==0:
        guess,guess_i = categoryFromOutput(output)
        correct = 'yes' if guess == category else 'NO (%s)'%category
        print('%d %d%% (%s) %.4f %s / %s %s' %(iter,iter/n_iters*100,timeSince(start),correct))
        
    if iter % plot_every == 0:
         all_losses.append(current_loss / plot_every)
         current_loss = 0
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.figure()
plt.plot(all_losses)
         
# Evaluating the Results

# keep track of correct guesses in a confusion matrix  
confusion = torch.zeros(n_categories,n_categories)
n_confusion = 10000

# just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output,hidden = rnn(line_tensor[i],hidden)
    return output

# go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category,line,category_tensor,line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess,guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] +=1
    
# normalize by dividing every row by its sum    
for i in range(n_categories):
    confusion[i]=confusion[i]/confusion[i].sum()

# set up plot
fig = plt.figure()
ax= fig.add_subplot(111)
cax = ax.matshow(confusion.numpy()) 
fig.colorbar(cax)

# set up axes
ax.set_xticklabels([''] + all_categories,rotation = 90) 
ax.set_yticklabels(['']+all_categories) 

# force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show() 
        
# running on user input
def predict(input_line,n_predictions=3):
    print('\n > %s' % input_line)
    output = evaluate(Variable(lineToTensor(input_line)))
    
    # get top N categories
    topv,topi = output.data.topk(n_predictioins,1,True)
    predictions=[]
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s'% (value,all_categories[category_index]))
        predictions.append([value,all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')






