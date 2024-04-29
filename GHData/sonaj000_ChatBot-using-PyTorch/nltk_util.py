import string
import nltk
import numpy as np
nltk.download("punkt") #trained tokenizer
from nltk.stem.porter import  PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence : string):
    return nltk.word_tokenize(sentence)

def stemming(word : string):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_setence, words):
    sentence_words = [stemming(word) for word in tokenized_setence]
    #initialize bag for 0 for each word
    bag = np.zeros(len(words),dtype= np.float32)
    for i, w, in enumerate(words):
        if w in tokenized_setence:
            bag[i] = 1.0
    
    return bag

