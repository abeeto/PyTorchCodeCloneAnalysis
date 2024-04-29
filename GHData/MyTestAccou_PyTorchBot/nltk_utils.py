import nltk
import numpy as np
# nltk.download('popular')

# from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer

snowball = SnowballStemmer(language='russian')


# stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence, language="russian")


def stem(word):
    return snowball.stem(word.lower())




def bag_of_words(tokenazed_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """

    tokenazed_sentence = [stem(word) for word in tokenazed_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenazed_sentence:
            bag[idx] = 1.0

    return bag


# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)






# words = ["Организация", "организовывать", "органический"]
# stemmed_words = [stem(w) for w in words]

# print(stemmed_words)

# a = "Добрый день могу я чем-то вам помоч ?"
# print(a)
# a = tokenize(a)
# print(a)