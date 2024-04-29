# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
MIN_WORD_FREQ = 1
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class TrainDataGenerator:
    def __init__(self, config, use_word=False):
        """
        :param use_word: word segment or char segment
        """
        self.config = config
        self.use_word = use_word

        '''build vocab'''
        print("LOAD OR BUILD VOCAB")
        if os.path.exists(config.vocab_path):
            self.vocab = pkl.load(open(config.vocab_path, 'rb'))
        else:
            self.vocab = self.build_vocab()
            pkl.dump(self.vocab, open(config.vocab_path, 'wb'))
        print(f"\nVOCAB SIZE: {len(self.vocab)}")

        '''build data set'''
        print("BUILD DATA SET")
        self.train_data, self.valid_data, self.test_data = self.build_data_set()

    def tokenizer(self, content):
        """
        TODO: MORE function
        segmentation function
        :param content: sentence
        :return:
        :rtype: list[str]
        """
        if self.use_word:
            token = content.split(' ')  # 以空格隔开，word-level
        else:
            token = [x for x in content]  # char-level
        return token

    def build_vocab(self):
        """
        build vocab
        :return:
        """

        vocab_freq_dic = {}
        with open(self.config.train_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc="BUILDING VOCAB"):
                line = line.strip()
                if not line:
                    continue
                content = line.split('\t')[0]
                token = self.tokenizer(content)
                for word in token:
                    vocab_freq_dic[word] = vocab_freq_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_freq_dic.items() if _[1] >= MIN_WORD_FREQ], key=lambda x: x[1],
                            reverse=True)[
                     :MAX_VOCAB_SIZE]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

        return vocab_dic

    def pad_sequence(self, token):
        seq_len = len(token)
        if self.config.pad_size:
            if seq_len < self.config.pad_size:
                token.extend([PAD] * (self.config.pad_size - len(token)))
            else:
                token = token[:self.config.pad_size]
        return token

    def load_data_set(self, path):
        """

        [([...], 0), ([...], 1), ...]
        :param path:
        :return:
        :rtype:List[Tuple[list, int, int]]
        """
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc="LOADING DATA"):

                line = line.strip()
                if not line:
                    continue

                content, label = line.split('\t')

                content_idx = []
                token = self.pad_sequence(self.tokenizer(content))
                for word in token:  # word to id
                    content_idx.append(self.vocab.get(word, self.vocab.get(UNK)))

                seq_len: int = self.config.pad_size
                contents.append((content_idx, int(label), seq_len))
        return contents

    def build_data_set(self):
        """

        :return:
        """
        train = self.load_data_set(self.config.train_path)
        dev = self.load_data_set(self.config.dev_path)
        test = self.load_data_set(self.config.test_path)
        return train, dev, test


class DataSetIterator(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        _x = [_[0] for _ in data]
        _y = [_[1] for _ in data]
        _seq_len = [_[2] for _ in data]

        x = torch.LongTensor(_x).to(self.device)
        y = torch.LongTensor(_y).to(self.device)
        seq_len = torch.LongTensor(_seq_len).to(self.device)

        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(data_set, config):
    _iter = DataSetIterator(data_set, config.batch_size, config.device)
    return _iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    # '''提取预训练词向量'''
    # # 下面的目录、文件名按需更改。
    # train_dir = "./THUCNews/data/train.txt"
    # vocab_dir = "./THUCNews/data/vocab.pkl"
    # pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    # emb_dim = 300
    # filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    # if os.path.exists(vocab_dir):
    #     word_to_id = pkl.load(open(vocab_dir, 'rb'))
    # else:
    #     # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    #     tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    #     word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #     pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    #
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    # f = open(pretrain_dir, "r", encoding='UTF-8')
    # for i, line in enumerate(f.readlines()):
    #     # if i == 0:  # 若第一行是标题，则跳过
    #     #     continue
    #     lin = line.strip().split(" ")
    #     if lin[0] in word_to_id:
    #         idx = word_to_id[lin[0]]
    #         emb = [float(x) for x in lin[1:301]]
    #         embeddings[idx] = np.asarray(emb, dtype='float32')
    # f.close()
    # np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    """TEST BUILD VOCAB"""
    # a = build_vocab('./THUCNews/data/eval.csv', lambda x: [y for y in x], 10000, 1)
    # from importlib import import_module
    #
    # __x = import_module('models.' + "TextCNN")
    # _config = __x.Config('THUCNews', "embedding_SougouNews.npz")
    # a = TrainDataGenerator(_config)
