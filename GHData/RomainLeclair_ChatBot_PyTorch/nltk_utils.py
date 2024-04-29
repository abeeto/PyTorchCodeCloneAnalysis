
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    Exemple
    setence = ["hello", "how", "are","you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [   0,       1  ,  0  ,  1  ,   0   ,   0   ,    0 ]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype= np.float32)
    for idx, w in enumerate(all_words):
        if w  in tokenized_sentence:
            bag[idx] = 1
    return bag


















a = "How long does shipping takes?"
#print(a)
a = tokenize(a)
#print(a)
words = ["organize", "organizes", "organizing"]
stemmed_words = [stem(w) for w in words]
#print(stemmed_words)