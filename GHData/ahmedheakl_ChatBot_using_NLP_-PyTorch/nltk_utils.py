# Tokenization
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# lower + Stemming


def stem(word):
    return stemmer.stem(word.lower())

# Bag of words


def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
