from torchtext.data import Dataset
from urllib.request import urlretrieve
from torchtext.data import Field
from torchtext.vocab import GloVe
from typing import Optional

import torchtext.data as data
import os

class dataset(Dataset):
    dirname = 'data/'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self,
                 text_field,
                 train='ptb.train.txt', test='ptb.test.txt', valid='ptb.valid.txt',
                 path=None, examples=None, **kwargs):

        fields = [('text', text_field)]

        path = self.dirname if path is None else path
        if not os.path.isdir(path):
            os.mkdir(path)

        url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'

        for file in [train, test, valid]:
            if not os.path.isfile(os.path.join(path, file)):
                print('Downloading {} data...'.format(file.split('.')[1]), end='')
                urlretrieve(url + file, os.path.join(path, file))
                print('Done')
            else:
                pass

        if examples is None:
            examples = []
            with open(path + '/' + train, errors='ignore') as f:
                examples += [data.Example.fromlist([line], fields) for line in f]
            with open(os.path.join(path, valid), errors='ignore') as f:
                examples += [data.Example.fromlist([line], fields) for line in f]
            with open(os.path.join(path, test), errors='ignore') as f:
                examples += [data.Example.fromlist([line], fields) for line in f]

        super(dataset, self).__init__(fields=fields, examples=examples, **kwargs)

    @classmethod
    def splits(cls,
               text_field,
               path=None,
               root='.data',
               **kwargs):

        examples = cls(text_field, **kwargs).examples
        train_index = 42068
        valid_index = 3370
        test_index = 3761

        return (cls(text_field, examples=examples[:train_index]),
                cls(text_field, examples=examples[train_index:(train_index + valid_index)]),
                cls(text_field, examples=examples[-test_index:]))

    @classmethod
    def ptb(cls,
            text_field,
            batch_size=16,
            device=-1,
            vector: Optional[str] = None,
            **kwargs):

        train, valid, test = cls.splits(text_field, **kwargs)
        train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test),
                                                                       batch_sizes=(batch_size, batch_size, batch_size),
                                                                       device=device,
                                                                       shuffle=True,
                                                                       repeat=True,
                                                                       sort_key=lambda x:len(x.text),
                                                                       **kwargs)
        if vector == 'glove_6B':
            vectors = GloVe('6B', dim=300)
        elif vector == 'glove_840B':
            vectors = GloVe('840B', dim=300)
        elif vector == 'glove_42B':
            vectors = GloVe('42B', dim=300)

        try:
            text_field.build_vocab(train, valid, test, vectors=vectors)
        except UnboundLocalError:
            print('No word embedding loaded.')
            text_field.build_vocab(train, valid, test)

        return (iter(train_iter), iter(valid_iter), iter(test_iter)), text_field

def load_data(word_emb, max_len):
    text_field = Field(init_token='<sos>',
                       eos_token='<eos>',
                       unk_token='<unk>',
                       pad_token='<pad>',
                       fix_length=max_len,
                       batch_first=True,
                       tokenize=lambda x: x.split())

    return dataset.ptb(text_field, vector=word_emb)

if __name__ == '__main__':
    TEXT = Field(init_token='<sos>',
                 eos_token='<eos>',
                 batch_first=True,
                 tokenize=lambda x:x.split())
    (train, valid, test), TEXT = dataset.ptb(TEXT)
    print(len(TEXT.vocab.stoi))
    print(len(train))
