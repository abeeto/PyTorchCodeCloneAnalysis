from collections import OrderedDict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean, cosine


class WordEmbedding(object):
    """
    Initialize with a text file containing one word per line, in the format
    <word> <vec0> <vec1> <vec2>...
    """
    def __init__(self, filename, n_neighbors=5, metric='minkowski'):

        with open(filename) as f:
            lines = f.read().split('\n')

        self.word_to_index = OrderedDict()
        self.index_to_word = OrderedDict()
        self.vectors = []
        index = 0
        for line in lines:
            if line:
                tokens = line.split()
                word = tokens[0]
                self.word_to_index[word] = index
                self.index_to_word[index] = word
                self.vectors.append(np.array(map(float, tokens[1:])))
                index += 1
        self.vectors = np.array(self.vectors)
        
        if metric == 'cosine':
            algo = 'brute'
        else:
            algo = 'auto'
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, 
                                     metric=metric, 
                                     algorithm=algo)
        self.nbrs.fit(self.vectors)

    @property
    def words(self):
        return self.word_to_index.keys()
    
    @property
    def dim(self):
        return self.vectors.shape[1]
    
    @property
    def shape(self):
        return self.vectors.shape
    
    def __contains__(self, word):
        return word in self.word_to_index.keys()
    
    def __len__(self):
        return len(self.vectors)
    
    def __str__(self):
        return 'WordEmbedding instance with %i words and %i-D vectors' % (len(self), self.dim)
    
    def __getitem__(self, word):
        if word not in self:
            raise KeyError('%s is not in the embedding vocabulary' % word)
        return self.vectors[self.word_to_index[word]]
    
    def dist(self, word1, word2, metric='euclidean'):
        word1_vec = self[word1]
        word2_vec = self[word2]
        if metric == 'euclidean':
            return euclidean(word1_vec, word2_vec)
        elif metric == 'cosine':
            return cosine(word1_vec, word2_vec)
        
    def nearest(self, input_, n_neighbors=1):
        """
        First arg is a word (str) or a vector (np.array), output is a word (str)
        """
        if isinstance(input_, np.ndarray):
            word_vec = input_
        else:
            word_vec = self[input_]
        dists, inds = self.nbrs.kneighbors(word_vec.reshape(1, -1))
        dists, inds = dists[0], inds[0]
        if dists[0] == 0:
            inds = inds[1:]
        n_neighbors = min([n_neighbors, self.nbrs.n_neighbors])
        return [self.index_to_word[ind] for ind in inds[:n_neighbors]]
