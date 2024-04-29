#%%
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
#%%
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]
#%%
# Preparation step:
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

def get_input(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

# Split sentences to tokens (i.e. separate words):
tokenized_corpus = tokenize_corpus(corpus)

# Get the vocabulary:
vocabulary = np.unique(sum(tokenized_corpus, [])).tolist()

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
#%%
# Define the skip-gram class:
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.lookup1 = nn.Linear(vocab_size, embedding_dim * vocab_size)
        self.lookup2 = nn.Linear(embedding_dim, vocab_size * embedding_dim)

    def forward(self, x, vocab_size, embedding_dim):
        x = torch.matmul(self.lookup1(x).view(embedding_dim, vocab_size), x)
        x = torch.matmul(self.lookup2(x).view(vocab_size, embedding_dim), x)
        x = F.log_softmax(x, dim=0).view(1,-1)
        return x
#%%
# Define a model and an optimizer:
embedding_dims = 5

# Define a model:
net = SkipGram(vocabulary_size, embedding_dims)

# Define optimizer:
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# Objective function:
criterion = nn.NLLLoss()
#%%
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make shure not jump out of sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
#%%
# Training loop:
num_epochs = 10000

for epoch in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        # compute input layer (center word encoded in one-hot manner)
        x = Variable(get_input(data)).float()
        
        # define real class
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        
        # reset the gradients back to zero
        optimizer.zero_grad()

        # compute output
        outputs = net(x, vocabulary_size, embedding_dims)
        
        # update loss
        train_loss = criterion(outputs, y_true)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the training loss to epoch loss
        loss_val += train_loss.item()

    if epoch % 500 == 0:    
        print(f'Loss at epo {epoch}: {loss_val/len(idx_pairs)}')
#%%
