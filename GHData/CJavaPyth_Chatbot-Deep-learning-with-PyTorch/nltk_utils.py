import nltk
nltk.download('punkt')      # a package with a pre-trained tokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):       # need to apply tokenization first
    """
    sentence = ['hello', 'how', 'are', 'you']
    words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']
    bag = [0, 1, 0, 1, 0 , 0, 0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

# sentence = ['hello', 'how', 'are', 'you']
# words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']
# bag = bag_of_words(sentence, words)
# print(bag)

# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

