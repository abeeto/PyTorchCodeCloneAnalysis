import re
import os
import random
import tarfile
import codecs
from torchtext import data
SEED = 1
import torch
torch.manual_seed(233)
random.seed(233)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class MR(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    train = []
    dev = []
    test = []

    def __init__(self, text_field, label_field, examples=None, **kwargs):

        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = "./datasets/data_raw/"
            with open(os.path.join(path, 'raw.clean.train'), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    sentence, flag = line.strip().split(' ||| ')
                    # emotion = 'positive' if flag == '1' else 'negative'
                    if flag == '0':
                        emotion = 'strong_negative'
                    elif flag == '1':
                        emotion = 'weak_negative'
                    elif flag == '3':
                        emotion = 'weak_positive'
                    elif flag == '4':
                        emotion = 'strong_positive'
                    elif flag == '2':
                        emotion = 'neutral'
                    self.train.append(data.Example.fromlist([sentence, emotion], fields))
            with open(os.path.join(path, 'raw.clean.dev'), 'r', encoding='utf-8') as f:
                for line in f:
                    sentence, flag = line.strip().split(' ||| ')
                    # emotion = 'positive' if flag == '1' else 'negative'
                    if flag == '0':
                        emotion = 'strong_negative'
                    elif flag == '1':
                        emotion = 'weak_negative'
                    elif flag == '3':
                        emotion = 'weak_positive'
                    elif flag == '4':
                        emotion = 'strong_positive'
                    elif flag == '2':
                        emotion = 'neutral'
                    self.dev.append(data.Example.fromlist([sentence, emotion], fields))
            with open(os.path.join(path, 'raw.clean.test'), 'r', encoding='utf-8') as f:
                for line in f:
                    sentence, flag = line.strip().split(' ||| ')
                    # emotion = 'positive' if flag == '1' else 'negative'
                    if flag == '0':
                        emotion = 'strong_negative'
                    elif flag == '1':
                        emotion = 'weak_negative'
                    elif flag == '3':
                        emotion = 'weak_positive'
                    elif flag == '4':
                        emotion = 'strong_positive'
                    elif flag == '2':
                        emotion = 'neutral'
                    self.test.append(data.Example.fromlist([sentence, emotion], fields))

            examples = self.train
        else:
            examples = examples
        super(MR, self).__init__(examples, fields, **kwargs)


    @classmethod
    def splits(cls, text_field, label_field, shuffle=True, root='.', **kwargs):

        examples = cls(text_field, label_field, **kwargs)    # type is class.MR object
        train_examples = examples.train
        dev_examples = examples.dev                                     # type is []
        test_examples = examples.test

        random.shuffle(train_examples)
        random.shuffle(dev_examples)
        random.shuffle(test_examples)
        print('train:', len(train_examples), 'dev:', len(dev_examples), 'test:', len(test_examples))
        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=dev_examples),
                cls(text_field, label_field, examples=test_examples),
                )

# load MR dataset
def load_mr(text_field, label_field, batch_size):
    print('loading data')
    train_data, dev_data, test_data = MR.splits(text_field, label_field)

    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    print('building batches')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data), batch_sizes=(batch_size, len(dev_data), len(test_data)), repeat=False, device=-1)

    return train_iter, dev_iter, test_iter
