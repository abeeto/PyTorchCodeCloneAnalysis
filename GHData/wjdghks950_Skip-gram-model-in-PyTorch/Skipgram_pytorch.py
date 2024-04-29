import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

# Tokenize the corpus
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(corpus)

vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)
            
# Create two dictionaries for mapping between word and index
word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

print(word_to_idx)
print(tokenized_corpus)


window_size = 2 # context_size
idx_pairs = []

# for each sentence
for sentence in tokenized_corpus:
    indices = [word_to_idx[w] for w in sentence]
    print(indices)
    # treat each word as a "center word"
    for center_word_pos in range(len(indices)):
        # window position based on the given window size
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            # keep an array of pairs of <center_word_idx, context_word_idx>
            idx_pairs.append((indices[center_word_pos], context_word_idx))
            
            # With each word as a "center word", check the context word positions
            print(idx_pairs)
            
idx_pairs = np.array(idx_pairs)


# Center word with one-hot encoding
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

# hidden layer
embedding_dim = 10
W1 = Variable(torch.randn(embedding_dim, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dim).float(), requires_grad=True)
epoch = 1000
lr = 0.001

for i in range(epoch):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data).float())
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        
        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
        
        # On top of the hidden layer, use "softmax layer"
        log_softmax = F.log_softmax(z2, dim=0)
        
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data[0]
        # Backpropagation
        loss.backward()
        
        W1.data -= lr * W1.grad.data
        W2.data -= lr * W2.grad.data
        
        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if i % 100 == 0:
        print("Epoch{}:".format(i), "Loss: {}".format(loss_val/len(idx_pairs)))