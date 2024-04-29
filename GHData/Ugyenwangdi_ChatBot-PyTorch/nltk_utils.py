# nltk framework for tokenization and stemming

import nltk
#nltk.download('punkt') # package with pretrained tokenizer

# For stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() # create stemmer
import numpy as np

# gets sentence
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)  # we need punkt

# gets word
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())  # stem the word and change to lower case

# gets all the words and clear, an array with all words
# apply tokenization
def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)    # Bag of words with same size of words with zeros
    for idx, w, in enumerate(all_words):  # gives both index and current word
        if w in tokenized_sentence:
            bag[idx] = 1.0      # as a float

    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "i", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)


# input string
# a = "How long does shipping take?"
# print(a)

# tokenize the string
# a = tokenize(a)
# print(a)

# # words for stemming
# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words] # use list comprehension
# print(stemmed_words)