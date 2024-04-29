import nltk
import torch
import random
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset


class Text8Dataset(Dataset):
    def __init__(self, filename, size, sub_sampling_min, sub_sampling_t, window_size,
                neg_sampling_pool_size, neg_sampling_size):
        self.word2idx = {}
        self.idx2word = {}
        self.window_size = window_size // 2
        self.neg_sampling_size = neg_sampling_size
        self.load_corpus(filename, size)
        self.sub_sampling(sub_sampling_min, sub_sampling_t, True, True)
        self.neg_sampling(neg_sampling_pool_size)

    def __len__(self): 
        return len(self.tokens)

    def __getitem__(self, idx):
        center = self.tokens[idx]
        context = self.tokens[max(0, idx-self.window_size):
                                min(idx+self.window_size+1, len(self.tokens))]
        context = [w for w in context if w != center]
        center = [center] * len(context)
        neg = self.get_neg_samples(center[0], len(center))
        return (center, context, neg)

    @staticmethod
    def collate_fn(batches):
        center_vec = torch.LongTensor(list(itertools.chain(*[batch[0] for batch in batches])))
        context_vec = torch.LongTensor(list(itertools.chain(*[batch[1] for batch in batches])))
        neg_mat = torch.LongTensor(np.vstack([batch[2] for batch in batches]))
        return center_vec, context_vec, neg_mat

    def load_corpus(self, filename, size):
        with open(filename) as f:
            self.tokens = list(f.readline().lower().split())[:size]
        self.num_words = len(self.tokens)
        self.word_set = set(self.tokens)
        self.word_counter = Counter(self.tokens)

    def sub_sampling(self, min_freq, t, sw: bool, sc: bool):
        """
            This method will filter out the following words:
                - disposable frequency words (< min_freq)
                - stopwords (the, in, of, ...)
                - single character words (a, aa, aaa, ...)
                - most frequent words
                - consecutive repeated words
        """
        df = pd.read_csv("wordsim353/combined.csv")
        wordsim353_words = set(df["Word 1"]) | set(df["Word 2"])
        if sw: stopwords = nltk.corpus.stopwords.words('english')
        for word, count in self.word_counter.items():
            if word not in wordsim353_words and \
            (count < min_freq or \
            (sw and word in stopwords) or \
            (sc and set(word) == set(word[0])) or \
            np.random.uniform() < 1-np.sqrt(t/(count/self.num_words))):
                self.word_set.discard(word)
        self.tokens = [token for token in self.tokens if token in self.word_set]
        self.tokens = [i[0] for i in itertools.groupby(self.tokens)]
        self.word_counter = Counter(self.tokens)
        for idx, word in enumerate(self.word_set):
            if word not in self.word2idx:
                self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.tokens = [self.word2idx[token] for token in self.tokens]

    def neg_sampling(self, neg_sampling_pool_size):
        self.neg_samples = []
        total = np.sum(np.power(np.array(list(self.word_counter.values())), 3/4))
        for word, count in self.word_counter.items():
            prob = np.power(count, 3/4) / total
            self.neg_samples += [self.word2idx[word]] * int(np.rint(prob * neg_sampling_pool_size))
        random.shuffle(self.neg_samples)

    def get_neg_samples(self, center, n):
        samples = np.random.choice(self.neg_samples, size=(n, self.neg_sampling_size))
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                while samples[i,j] == center:
                    samples[i,j] = np.random.choice(self.neg_samples, size=1)[0]
        return samples
