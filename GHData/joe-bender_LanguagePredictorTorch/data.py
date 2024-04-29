import re
import random

def get_lines(filename):
    text = None
    with open(filename, 'r', encoding='utf8') as f:
        text = f.readlines()
    return text

def get_vocab(lines):
    vocab = set()
    for line in lines:
        for word in line.split():
            if re.match('^[a-z]+$', word):
                vocab.add(word)
    return vocab

def remove_common_words(english_vocab, french_vocab, spanish_vocab):
    common_words = (english_vocab & french_vocab)\
        | (english_vocab & spanish_vocab) | (french_vocab & spanish_vocab)
    english_vocab = english_vocab - common_words
    french_vocab = french_vocab - common_words
    spanish_vocab = spanish_vocab - common_words
    return english_vocab, french_vocab, spanish_vocab

def trim_vocabs(english_vocab, french_vocab, spanish_vocab):
    end = min(len(english_vocab), len(french_vocab), len(spanish_vocab))
    english_vocab = set(list(english_vocab)[:end])
    french_vocab = set(list(french_vocab)[:end])
    spanish_vocab = set(list(spanish_vocab)[:end])
    return english_vocab, french_vocab, spanish_vocab

def vocabs_to_dataset(english_vocab, french_vocab, spanish_vocab):
    english_dataset = [(word, 'english') for word in english_vocab]
    french_dataset = [(word, 'french') for word in french_vocab]
    spanish_dataset = [(word, 'spanish') for word in spanish_vocab]
    dataset = english_dataset + french_dataset + spanish_dataset
    return dataset

def get_dataset():
    english_lines = get_lines('text/english.txt')
    french_lines = get_lines('text/french.txt')
    spanish_lines = get_lines('text/spanish.txt')
    # trim the meta-text from the beginning and end (because it's in english in all of the files)
    english_lines = english_lines[34:1899]
    french_lines = french_lines[36:2189]
    spanish_lines = spanish_lines[35:2009]
    # get vocabs
    english_vocab = get_vocab(english_lines)
    french_vocab = get_vocab(french_lines)
    spanish_vocab = get_vocab(spanish_lines)
    # remove all common words, making each vocabulary unique
    english_vocab, french_vocab, spanish_vocab =\
        remove_common_words(english_vocab, french_vocab, spanish_vocab)
    # make all vocabs the same size
    english_vocab, french_vocab, spanish_vocab =\
        trim_vocabs(english_vocab, french_vocab, spanish_vocab)
    dataset = vocabs_to_dataset(english_vocab, french_vocab, spanish_vocab)
    return dataset

def get_pairs(filename):
    pairs = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            pair = line
            pair = pair.rstrip()
            pair = pair.split(',')
            pair = (pair[0], pair[1])
            pairs.append(pair)
    return pairs
