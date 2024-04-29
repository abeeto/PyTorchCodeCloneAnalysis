import collections
import numpy as np


"""
part 1: TF, IDF, TF-IDF
"""

class IDF:
    """
    Inverse document frequency
    see https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html
    """
    def __init__(self, corpus):
        """
        corpus is a iterable
        each item is a collection of strings
        """
        word2df = collections.defaultdict(int)
        N = 0
        for doc in corpus:
            for word in set(doc):
                word2df[word] += 1
            N += 1
        self.N = N
        self._idf = {
            word: np.log(N / v) for word, df in word2df.items()
        }
    
    def get(self, word, default=None):
        """
        get idf for word, if doesnt exist, count it as 1
        """
        if default is None:
            default = np.log(self.N)
        return self._idf.get(word, default)


class TF:
    """
    Term frequency
    see https://nlp.stanford.edu/IR-book/html/htmledition/term-frequency-and-weighting-1.html
    """
    def __init__(self, doc):
        word2count = collections.defaultdict(int)
        N = 0
        for word in doc:
            word2count[word] += 1
            N += 1
        self._tf = {
            word: count / N for word, count in word2count.items()
        }
    
    def get(self, word, default=None):
        if default is None:
            default = 0.0
        return self._tf.get(word, default)


class TFIDF:
    """
    Term frequency - inverse document frequency
    see https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html
    """
    def __init__(self, idf, tf):
        self.idf = idf
        self.tf = tf
    
    def get(self, word, tf_default=None, idf_default=None):
        return self.tf.get(word, tf_default) * self.idf.get(word, idf_default)