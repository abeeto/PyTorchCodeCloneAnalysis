import torch
import torch.nn as nn
import re

def get_words(filename):
    with open(filename, 'r', encoding='utf8') as f:
        text = f.read()

        # make all words lowercase so capitalization isn't a factor
        text = text.lower()
        # remove all punctuation, except apostrophes in contractions
        text = re.sub('!|,|;|\(|\)|\.|‘|’\W|\*|\n|-|:|\[|\]|\?|_|“|”|\ufeff', ' ', text)
        # turns all sequences of white space into a single space to separate words
        text = re.sub('\s+', ' ', text)

        words = text.split()
        return words

def get_word_mappings(words):
    vocab = sorted(list(set(words)))
    word_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_word = {i: word for i, word in enumerate(vocab)}
    assert(len(word_to_int) == len(int_to_word))
    return word_to_int, int_to_word

def get_training_pairs(words, window_size=5):
    pairs = []
    for i in range(0, len(words)):
        left = max(0, i-window_size)
        right = min(len(words)-1, i+window_size)
        left_range = words[left:i]
        right_range = words[i+1:right+1]
        targets = left_range + right_range
        input = words[i]
        for target in targets:
            pairs.append((input, target))
    return pairs

def index_to_1hot(index, size):
    tensor = torch.zeros(size)
    tensor[index] = 1
    return tensor

def index_from_1hot(tensor):
    return tensor.argmax().item()
