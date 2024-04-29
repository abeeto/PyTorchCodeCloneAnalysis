import torch
import torchtext
from collections import Counter, defaultdict

# things we need an nlp preprocessing pipeline to do:

# Train/val/test split: 
# File Loading:
# Tokenization:
# Vocab:
# Numericalize/ Indexify: 
# Word Vector:
# Batching:
# Embedding Lookup:


# 1: tokenize
sentence = "The quick brown fox jumped over the lazy dog"
sentence = sentence.lower()
sentence = sentence.split(" ")

some_sentences = [sentence]


# 2: vocab:
vocab = set(tokens)

# create counter of words in corpus 
freq = Counter([word for sentence in some_sentences for word in sentence])
freq

# print the 3 most common vocab words in token list
freq.most_common(3)


# get vocab and most frequent:
max_vocab = 60000
min_freq = 1


# return top most common words, max_vocab and words that
                    # appear more than the min threshold. 
itos = [word for word, value in freq.most_common(max_vocab)
                                   if value >= min_freq]
itos


# 3: Numericalize

# create string to integer mapping:
stoi = defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

stoi

import numpy as np

# create string to integer mapping
nums = np.array([[stoi[word] for word in sentence] for sentence in 
                                                    some_sentences])

nums 


