import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        # return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def one_hot(seq_batch, depth):
    out = torch.zeros(seq_batch.size()+torch.Size([depth]))
    return out.scatter_(2, seq_batch.view(seq_batch.size()+torch.Size([1])), 1)

class Corpus(object):
    def __init__(self, path, device, batch_size=8, seq_len=32):
        self.dictionary = Dictionary()
        self.batch_size = batch_size
        self.seq_len = seq_len

        #### train data ####
        train_data = self.tokenize_data(os.path.join(path, 'pku_train/pku_no_space.txt'))
        self.train_length = train_data.size(0)
        train_label = self.tokenize_label(os.path.join(path, 'pku_train/pku_label.txt'), self.train_length)

        train_nbatch = train_data.size(0) // self.batch_size
        self.total_num_of_train_batches = train_nbatch // self.seq_len

        self.train_data_batched = self.batchify(train_data, train_nbatch).to(device)
        self.train_label_batched = self.batchify(train_label, train_nbatch).to(device)

        #### test data ####
        test_data = self.tokenize_data(os.path.join(path, 'pku_test/pku_no_space.txt'))
        self.test_length = test_data.size(0)
        test_label = self.tokenize_label(os.path.join(path, 'pku_test/pku_label.txt'), self.test_length)

        test_nbatch = test_data.size(0) // self.batch_size
        self.total_num_of_test_batches = test_nbatch // self.seq_len

        self.test_data_batched = self.batchify(test_data, test_nbatch).to(device)
        self.test_label_batched = self.batchify(test_label, test_nbatch).to(device)

    def tokenize_data(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = list(line) #+ ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = list(line) #+ ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def tokenize_label(self, path, length):
        with open(path, 'r', encoding='utf8') as f:
            ids = torch.LongTensor(length)
            index = 0
            for line in f:
                labels = list(line)
                for label in labels:
                    if label == '\n':
                        ids[index] = 2
                    else:
                        ids[index] = int(label)
                    index += 1
        return ids

    def batchify(self, data, nbatch):
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        data = data.view(self.batch_size, -1).t().contiguous()
        return data

    def get_train_batch(self, i):
        return self.train_data_batched[i * self.seq_len:(i + 1) * self.seq_len],\
               self.train_label_batched[i * self.seq_len:(i + 1) * self.seq_len]

    def get_test_batch(self, i):
        return self.test_data_batched[i * self.seq_len:(i + 1) * self.seq_len],\
               self.test_label_batched[i * self.seq_len:(i + 1) * self.seq_len]
