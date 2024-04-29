import torch.utils.data as data

import os
import re
import torch
import pdb
import random
import pickle
# import attacker
from collections import defaultdict
import numpy as np
import torch
from process import construct_dataset, create_labels


root = '/data/yanjianhao/nlp/torch/torch_NRE/data/'


def set_default():
    return None


def preprocess(file_path, w_to_ix):
    rel_to_ix = defaultdict(set_default)
    dict = defaultdict(list)
    en_to_rel = defaultdict(set_default)
    dict_text = defaultdict(list)
    # set NA
    rel_to_ix['NA'] = 0

    with open("/data/yanjianhao/nlp/torch/torch_NRE/origin_data/relation2id.txt", 'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.split()
        rel_to_ix[l[0]] = int(l[1])

    with open(file_path, 'r') as f:
        lines = f.readlines()

    max_len = 0
    for line in lines:
        splits = line.split()
        en1 = splits[2]
        en2 = splits[3]
        rel = splits[4]
        rel = rel_to_ix[rel] if rel_to_ix[rel] is not None else rel_to_ix['NA']
        # construct rel_to_ix
        # if rel_to_ix[rel] == None:
        #     rel_to_ix[rel] = len(rel_to_ix) - 1
        # rel = rel_to_ix[rel]


        # construct en_to_rel
        if not en_to_rel[(en1, en2)]:
            en_to_rel[(en1, en2)] = rel
            en_to_rel[(en2, en1)] = rel

        tmp = splits[5:-1]
        # keep the max_length for a sentence
        if max_len < len(tmp):
            max_len = len(tmp)
        en1_pos = 0
        en2_pos = 0
        found_1 = found_2 = 0

        for i in range(len(tmp)):
            if tmp[i] == en1:
                en1_pos = i
                found_1 = 1
            if tmp[i] == en2:
                en2_pos = i
                found_2 = 1
            if found_2 & found_1:
                break

        # sent = " ".join(tmp)
        dict_text[(en1, en2)].append(" ".join(tmp))
        tmp = [w_to_ix[word] if w_to_ix[word] else w_to_ix['UNK'] for word in tmp]
        dict[(en1, en2)].append((en1_pos, en2_pos, tmp))
    # save intermediate results
    with open("/data/yanjianhao/nlp/torch/torch_NRE/origin_data/relaton_to_ix.txt", 'w') as f:
        lines = []
        for key, value in rel_to_ix.items():
            lines.append(str(key) + " " + str(value) + "\n")
        f.writelines(lines)

    return dict, en_to_rel, rel_to_ix, max_len, dict_text


# w2v --->   word : ndarray
def read_in_vec(path):
    D = 50
    w_to_ix = defaultdict(set_default)
    with open(path, 'r')as f:
        lines = f.readlines()
    vecs = []
    for ix, line in enumerate(lines[1:]):
        splits = line.split()
        # convert text to numpy array
        vec = np.array([float(n) for n in splits[1:]])
        if w_to_ix[splits[0]]:
            pdb.set_trace()
        w_to_ix[splits[0]] = ix
        vecs.append(vec)
    w_to_ix['UNK'] = len(vecs)
    w_to_ix['<t>'] = len(vecs) + 1
    vecs.append(np.random.randn(D))
    vecs.append(np.random.randn(D))

    # vecs -> vocab_size * D
    vecs = np.stack(vecs, axis=1).T
    return w_to_ix, vecs


# def add_position(dict):


class Dataset(data.Dataset):
    def __init__(self, root, train_test='train', transform=None, position_embed=True):
        if train_test == 'train':
            file_name = 'train.txt'
        else:
            file_name = 'test.txt'
        vec_name = 'vec.txt'
        self.w_to_ix, self.vecs = read_in_vec(os.path.join(root, vec_name))
        assert len(self.w_to_ix) == self.vecs.shape[0]
        # dict : [(en1_pos, en2_pos, [word,]),]
        self.dict, self.en_to_rel, self.rel_to_ix, self.max_sent_len, self.dict_text = preprocess(os.path.join(root, file_name), self.w_to_ix)
        self.key_list = list(self.dict.keys())
        self.vocab_size = self.vecs.shape[0]
        # self.n_rel = len(self.rel_to_ix)
        self.n_rel = 53
        # if position_embed:

        print("Vocab_size is:")
        print(self.vocab_size)
        print("Relation number is:")
        print(self.n_rel)
        print("# of bags:")
        print(self.__len__())

    def __getitem__(self, index):
        # multi-instance learning
        # using bag as inputs
        # en_pair = list(self.dict.keys())[index]
        en_pair = self.key_list[index]
        bag = self.dict[en_pair]
        target = self.en_to_rel[en_pair]
        # print(en_pair)
        if target == None:
            pdb.set_trace()
        return bag, target


    def __len__(self):
        return len(self.dict.keys())


def collate_fn(data):
    '''

    :param data: list -> (pos1, pos2, [words,]), label
    :return: bags, label
             bags : pos1, pos2, sents

    '''
    labels = torch.LongTensor([item[1] for item in data])
    bags = [item[0] for item in data]

    return bags, labels


def collate_fn_temporal(data):
    return


class Temporal_Data(data.Dataset):
    def __init__(self, root, train_test='train', transform=None, position_embed=True):
        if train_test == 'train':
            file_name = 'train_temporal.txt'
        else:
            file_name = 'test_temporal.txt'
        vec_name = 'vec.txt'
        self.w_to_ix, self.vecs = read_in_vec(os.path.join(root, vec_name))
        assert len(self.w_to_ix) == self.vecs.shape[0]
        # dict : [(en1_poss, en2_pos, [word,]),]

        # self.dict, self.en_to_rel, self.rel_to_ix, self.max_sent_len = preprocess(os.path.join(root, file_name),
                                                                                  # self.w_to_ix)
        self.labels = create_labels()
        self.dict, self.rel_to_ix = construct_dataset(os.path.join(root, file_name), self.labels, self.w_to_ix)
        self.key_list = list(self.dict.keys())

        self.vocab_size = self.vecs.shape[0]
        self.n_rel = len(self.rel_to_ix)
        # if position_embed:

        print("Vocab_size is:")
        print(self.vocab_size)
        print("Relation number is:")
        print(self.n_rel)
        print("# of bags:")
        print(self.__len__())

    def __getitem__(self, index):
        # multi-instance learning
        # using bag as inputs

        en_pair = list(self.key_list)[index]
        # there are Mention objects in the bag
        bag = self.dict[en_pair]
        target = [item.tag for item in bag]
        return bag, target

    def __len__(self):
        return len(self.key_list)



if __name__ == "__main__":
    root = '/data/yanjianhao/nlp/torch/torch_NRE/data/'
    dataset = Dataset(root)



