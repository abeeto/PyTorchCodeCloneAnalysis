import codecs
import numpy as np
import re
from typing import Tuple, List, Dict

import model
import utils


def load_sentences(path: str) -> List[List[List[str]]]:
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = re.sub(r'[0-9]', '0', line.rstrip())
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()  # List[str]
            assert len(word) >= 2
            sentence.append(word)  # List[List[str]]
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)  # List[List[List[str]]]

    utils.update_tag_scheme(sentences, 'iobes')  # iobes or iob
    return sentences


def word_mapping(sentences: List[List[List[str]]]):
    words = [[x[0] for x in s] for s in sentences]
    dico = utils.create_dico(words)
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k: v for k, v in dico.items() if v >= 3}
    word_to_id, id_to_word = utils.create_desc_mapping(dico)
    print("Found {} unique words ({} in total).".format(len(dico), sum(len(x) for x in words)))
    return dico, word_to_id, id_to_word


def char_mapping(sentences: List[List[List[str]]]):
    chars: List[List[str]] = ["".join([w[0] for w in s]) for s in sentences]  # List[List[char]]
    dico = utils.create_dico(chars)
    dico['<PAD>'] = 10000000
    char_to_id, id_to_char = utils.create_desc_mapping(dico)
    print("Found {} unique characters.".format(len(dico)))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences: List[List[List[str]]]):
    tags = [[word[-1] for word in s] for s in sentences]
    dico = utils.create_dico(tags)
    dico[model.SOS_TAG] = -1
    dico[model.EOS_TAG] = -2
    tag_to_id, id_to_tag = utils.create_desc_mapping(dico)
    print("Found {} unique named entity tags.".format(len(dico)))
    return dico, tag_to_id, id_to_tag


def load_pretrained_embedding(dictionary: Dict[str, int], pretrained_path: str, word_dim: int):
    print('Loading pretrained embedding from {}...'.format(pretrained_path))
    lines = list(codecs.open(pretrained_path, 'r', 'utf-8'))
    pretrained = set([line.strip().split()[0].strip() for line in lines])
    for word in pretrained:
        if word not in dictionary:
            dictionary[word] = 0
    word_to_id, id_to_word = utils.create_desc_mapping(dictionary)

    all_word_embedding = {}
    for line in lines:
        s = line.strip().split()
        if len(s) == word_dim + 1:
            all_word_embedding[s[0]] = np.array([float(i) for i in s[1:]])
    word_embedding: np.ndarray = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))
    for w in word_to_id:
        if w in all_word_embedding:
            word_embedding[word_to_id[w]] = all_word_embedding[w]
        elif w.lower() in all_word_embedding:
            word_embedding[word_to_id[w]] = all_word_embedding[w.lower()]
    print('Loaded {} pretrained embedding.'.format(len(all_word_embedding)))

    return dictionary, word_to_id, id_to_word, word_embedding


class Data:
    """
    Single line of data in dataset.
    """

    def __init__(self,
                 str_words: List[str], words: List[int],
                 chars: List[List[int]], chars_mask: List[List[int]], chars_length: List[int], chars_d: Dict[int, int],
                 caps: List[int], tags: List[int]):
        self.str_words = str_words
        self.words = words
        self.chars = chars
        self.chars_mask = chars_mask
        self.chars_length = chars_length
        self.chars_d = chars_d
        self.caps = caps
        self.tags = tags
        self.words_prefix_ids: List[List[int]] = []
        self.words_suffix_ids: List[List[int]] = []


def prepare_dataset(sentences: List[List[List[str]]], word_to_id: Dict[str, int], char_to_id: Dict[str, int], tag_to_id: Dict[str, int]) -> List[Data]:
    dataset: List[Data] = []
    for sentence in sentences:
        str_words = [word[0] for word in sentence]
        words = [word_to_id[word if word in word_to_id else '<UNK>'] for word in str_words]
        chars = [[char_to_id[char] for char in word if char in char_to_id] for word in str_words]  # Skip characters that are not in the training set
        chars_mask, chars_length, chars_d = utils.align_char_lists(chars)
        caps = [model.cap_feature(word) for word in str_words]
        tags = [tag_to_id[word[-1]] for word in sentence]

        data = Data(str_words=str_words, words=words, chars=chars, chars_mask=chars_mask, chars_length=chars_length, chars_d=chars_d, caps=caps, tags=tags)
        dataset.append(data)
    return dataset


def add_affix_to_datasets(train_data: List[Data], val_data: List[Data], test_data: List[Data]):
    print('Generating affix list from training dataset...')
    prefix_dicts, suffix_dicts = utils.get_affix_dict_list([data.str_words for data in train_data], threshold=125)

    for dataset in [train_data, val_data, test_data]:
        for data in dataset:
            words_prefix_ids, words_suffix_ids = utils.get_words_affix_ids(data.str_words, prefix_dicts, suffix_dicts)
            data.words_prefix_ids = words_prefix_ids  # List[List[int]], dim0: words, dim1: n-grams (# = 4), value: id (>=0)
            data.words_suffix_ids = words_suffix_ids

    return prefix_dicts, suffix_dicts
