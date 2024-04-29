import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
#nltk.download('punkt')

stemmer = PorterStemmer()

def tokenise(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenised_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag_of_words = [0, 1, 0, 1, 0, 0, 0]
    """
    # apply stemming
    tokenised_sentence = [stem(w) for w in tokenised_sentence]

    # find matching words in tokenised sentence and all_words; place 1s in corresponding places in bag
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w, in enumerate(all_words):
        if w in tokenised_sentence:
            bag[i] = 1

    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# print(bag_of_words(sentence, words))

# a = "How long does shipping take?"
# print(a)
# a = tokenise(a)
# print(a) 
# words = ["Organize", "organizing", "Organizes"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)