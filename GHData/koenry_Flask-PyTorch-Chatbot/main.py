import numpy as np
import nltk
# nltk.download('punkt') 
# might need to download 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence) 

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenizedSentence, allWords):
    pass
    tokenizedSentence = [stem(i) for i in tokenizedSentence]
    bag = np.zeros(len(allWords), dtype=np.float32)
    for idx, i in enumerate(allWords):
        if i in tokenizedSentence:
            bag[idx] = 1.0
    return bag
   
