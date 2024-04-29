# -*- coding: utf-8 -*-
"""
Word Embeddings: Encoding Lexical Semantics
using N-gram models and continuous bag of words (CBOW) as 
a practice example for excercising with PyTorch
"""


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import re as re
import os


torch.manual_seed(1)

######################################################################

def getText(filepath):
    fileText = None
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        fileText = """The US media reports suggest Robert Mueller's inquiry has taken the first step towards possible criminal charges.
        According to Reuters news agency, the jury has issued subpoenas over a June 2016 meeting between President Donald Trump's son and a Russian lawyer.
        The president has poured scorn on any suggestion his team colluded with the Kremlin to beat Hillary Clinton.
        In the US, grand juries are set up to consider whether evidence in any case is strong enough to issue indictments for a criminal trial. They do not decide the innocence or guilt of a potential defendant.
        The panel of citizens also allow a prosecutor to issue subpoenas, a legal writ, to obtain documents or compel witness testimony under oath."""
        
    with open('filepath','r') as ff:
        fileText = ff.read()
    fileText = re.sub('[?!,.-_;:$%@\(\)\"\']','',fileText)
    return fileText

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# building a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# printing the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.ct_sz = context_size
        self.word_to_ix = {}

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs
    
    def makeNgramsSquences(self,text):
        n_grams=[]
        for i in range(len(text) - self.ct_sz):
            n_gram=[text[i+j] for j in range(self.ct_sz)]
            n_grams.append((n_gram,text[i+self.ct_sz]))
        vocab = set(text)
        word_to_ix = {word: i for i,word in enumerate(vocab)}
        self.word_to_ix.update(word_to_ix)
        return n_grams
    
    def train(self,text,loss_fn=nn.NLLLoss(),optimizer=optim.SGD(self.parameters(), lr=0.001),epoch=10):
        losses = []
        n_grams = self.makeNgramsSquences(text)
        for ep in range(epoch):
            total_loss = torch.Tensor([0])
            for context, target in n_grams:

                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in variables)
                context_idxs = [self.word_to_ix[w] for w in context]
                context_var = autograd.Variable(torch.LongTensor(context_idxs))
        
                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                self.zero_grad()
        
                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = self.forward(context_var)
        
                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a variable)
                loss = loss_fn(log_probs, autograd.Variable(
                    torch.LongTensor([self.word_to_ix[target]])))
        
                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()
        
                total_loss += loss.data
            losses.append(total_loss)
        return losses


model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
print(losses)  # The loss decreased every iteration over the training data!


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.linear1 = nn.Linear(embedding_dim,128)
        self.linear2 = nn.Linear(128,vocab_size)
        self.csSize = context_size

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embedsVec = autograd.Variable(torch.zeros(embeds.size(1)))
        for i in range(embeds.size(0)):
            embedsVec += embeds[i]
        embedsVec = embedsVec.view(1,-1)
        out = F.relu(self.linear1(embedsVec))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs
        

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


make_context_vector(data[0][0], word_to_ix)  # example

losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size,EMBEDDING_DIM,2*CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context,target in data:
        model.zero_grad()
        contextVT_ind = make_context_vector(context,word_to_ix)
        lgpb = model(contextVT_ind)
        loss = loss_function(lgpb,autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        
    losses.append(total_loss)
print (losses)