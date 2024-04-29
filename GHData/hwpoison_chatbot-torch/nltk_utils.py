import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer() 
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentences, all_words):
    sentence_words = [stem(word) for word in tokenized_sentences]
    bag = np.zeros(len(all_words), dtype=np.float32)                                    
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0 
    return bag
