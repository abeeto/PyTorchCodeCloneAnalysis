import os
import nltk
import pickle

from pycocotools.coco import COCO
from config import hparams


class Vocabulary:
    """
    词表类
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[Vocabulary.unknown_token()]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def start_token():
        return '<start>'

    @staticmethod
    def end_token():
        return '<end>'

    @staticmethod
    def unknown_token():
        return '<unk>'

    @staticmethod
    def padded_token():
        return '<pad>'


def build_vocab(cap_path, threshold=5):
    """
    按照训练集的caption建立词表

    :param cap_path:
    :param threshold:
    :return:
    """
    vocab = Vocabulary()
    vocab.add_word(Vocabulary.padded_token())
    vocab.add_word(Vocabulary.start_token())
    vocab.add_word(Vocabulary.end_token())
    vocab.add_word(Vocabulary.unknown_token())
    for word in get_filtered_words(cap_path, threshold):
        vocab.add_word(word)

    print('Total %d words in vocabulary.' % len(vocab))
    return vocab


def get_filtered_words(cap_path, threshold=5):
    """
    按照词表长最大值，出现次数最小值过滤词汇

    :param cap_path:
    :param threshold:
    :return:
    """
    from collections import Counter
    coco = COCO(cap_path)
    ann_ids = coco.anns.keys()
    counter = Counter()
    for idx, ann_id in enumerate(ann_ids):
        cap = str(coco.anns[ann_id]['caption'])
        tokens = nltk.tokenize.word_tokenize(cap.lower())
        counter.update(tokens)
        if idx % 10000 == 0:
            print('[%d/%d] Tokenized the captions.' % (idx, len(ann_ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    return words


def dump_vocab(vocab_pkl, cap_path, threshold=5):
    """
    持久化词表到vocab.pkl

    :param vocab_pkl: 词表保存的路径
    :param cap_path: caption用于建立词表
    :param threshold: 高于threshold的词汇才被加入词表
    :return:
    """
    print('*' * 20, 'dump vocab', '*' * 20)
    if not os.path.exists(vocab_pkl):
        vocab = build_vocab(cap_path, threshold)
        with open(vocab_pkl, 'wb') as f:
            pickle.dump(vocab, f)

        print('Total vocabulary size: %d' % len(vocab))
        print('Saved the vocabulary to: %s' % vocab_pkl)
    else:
        print('Vocabulary already exists.')


def load_vocab(vocab_pkl):
    """
    从vocab_pkl读取保存的词表

    :param vocab_pkl:
    :return:
    """
    print('*' * 20, 'load vocab', '*' * 20)
    with open(vocab_pkl, 'rb') as f:
        return VocabularyUnpickler(f).load()


class VocabularyUnpickler(pickle.Unpickler):
    """
    辅助类，解决pickle.load()反序列化找不到类定义的问题
    """
    def find_class(self, module, name):
        if name == 'Vocabulary':
            return Vocabulary
        return super().find_class(module, name)


if __name__ == '__main__':
    print('vocab.pkl: ', hparams.vocab_pkl)
    print('cap_path:', hparams.train_cap)
    dump_vocab(vocab_pkl=hparams.vocab_pkl, cap_path=hparams.train_cap)
    vocabulary = load_vocab(hparams.vocab_pkl)
    for i in range(600, 650):
        print(vocabulary.idx2word[i])
