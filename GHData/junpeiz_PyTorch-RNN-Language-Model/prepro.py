from os import path


class Dict(object):
    def __init__(self):
        self.w2idx = dict()
        self.idx2w = []

    def add_word(self, word):
        if word not in self.w2idx:
            self.w2idx[word] = len(self.idx2w)
            self.idx2w.append(word)

    def __len__(self):
        return len(self.idx2w)


class Corpus(object):
    def __init__(self, opt):
        self.dictionary = Dict()
        train_file = 'train_small.txt' if opt.debug else 'train.txt'
        val_file = 'valid_small.txt' if opt.debug else 'valid.txt'
        test_file = 'test_small.txt' if opt.debug else 'test.txt'
        self.train_data = self.tokenize(path.join(opt.data_dir, train_file))
        self.val_data = self.tokenize(path.join(opt.data_dir, val_file))
        self.test_data = self.tokenize(path.join(opt.data_dir, test_file))

    def tokenize(self, file_path):
        assert path.exists(file_path)
        count = 0

        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip() == '' or line.strip()[0] == '=':
                    continue
                tokens = line.strip().split() + ['<eos>']
                count += len(tokens)
                for token in tokens:
                    self.dictionary.add_word(token)

        target = [0] * count
        count = 0
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip() == '' or line.strip()[0] == '=':
                    continue
                tokens = line.strip().split() + ['<eos>']
                for token in tokens:
                    target[count] = self.dictionary.w2idx[token]
                    count += 1
        return target


def get_data(opt):
    return Corpus(opt)
