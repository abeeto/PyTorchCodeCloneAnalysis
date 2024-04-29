#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example embedding look-up for explaining how torch.index_select and torch.view work
Inspired by assignment 3 of Stanford's CS224n class on NLP, spring 2020
For explanations, see dstei.github.io/pytorch-index_select-example
"""

import torch
torch.manual_seed(0)

batch_size = 2
no_features = 3
embedding_size = 5
num_words = 100

words = torch.randint(100, (batch_size, no_features)) # Initialise w with random words
words[:, 0] = torch.arange(batch_size) # This is to mark/recognise the head of each row of words
print('words', '\n', words, '\n')

embeddings = torch.rand(num_words, embedding_size) # Initialise embeddings with random word vectors
embeddings[:, 0] = torch.arange(num_words) # This is to mark/recognise the head of each row of embeddings
print('embeddings[:10]', '\n', embeddings[:10], '\n')

words_reshaped = words.view(batch_size*no_features)
print('words_reshaped', '\n', words_reshaped, '\n')
selected_embeddings = torch.index_select(embeddings, 0, words_reshaped)
print('selected_embeddings', '\n', selected_embeddings, '\n')
embedded_words = selected_embeddings.view(batch_size, no_features*embedding_size)
print('selected_embeddings reshaped', '\n', embedded_words, '\n')

embedded_words = torch.index_select(embeddings, 0, words.view(-1)).view(-1, no_features*embedding_size) #fully vectorised, i.e. fast to compute
print('one-liner', '\n', embedded_words, '\n')
