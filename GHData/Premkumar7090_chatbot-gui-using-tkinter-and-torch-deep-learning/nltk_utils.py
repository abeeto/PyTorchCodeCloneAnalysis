import nltk
import numpy as np

# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokanized_sentence, all_words):
    sent = ["a","b", "c", "d"]
    word = [0,1,0,0]
    tokanized_sentence = [stem(w) for w in tokanized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokanized_sentence:
            bag[idx] = 1.0
    return bag

sentence = ["a","b","c","d"]
words = ["e","a","f","d"]
bag = bag_of_words(sentence, words)
print(bag)

# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["organize","organizes","organizing"]
# ste = [stem(word) for word in words]
# print(ste)