# coding: utf-8

import codecs
import random
from configurations import get_conf
import torch
import os
import numpy

def pad_tgt_seq(seq, max_length):
    # max_length = 270
    pad = numpy.zeros((max_length - seq.size), dtype='int32')
    pad_seq = numpy.hstack((seq, pad))
    return pad_seq

def pad_src_seq(seq, max_length):
    # max_length = 225
    pad = numpy.zeros((max_length-seq.shape[0], seq.shape[1]), dtype='int32')
    pad_seq = numpy.vstack((seq, pad))
    return pad_seq




def get_source_mask(batch, hidden_size, max_length, lengths):
    mask = torch.zeros(batch, max_length, hidden_size)
    for i in range(batch):
        mask[i, :lengths[i], :] = 1
    return mask.transpose(0, 1)

def get_target_mask(batch, max_length, lengths):
    mask = torch.zeros(batch, max_length)
    for i in range(batch):
        mask[i, :lengths[i]] = 1
    return mask.transpose(0, 1)






class BatchGetter(object):
    def __init__(self, file_name, batch_size, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._vocabs = config['Vocabs']
        self._outtag_voc = config['OutTags']
        self._fea_pos = config['fea_pos']
        self._word_pos = config['WordPos']

        self._vocab_char = config['CharVoc']
        self._max_char_len = config['max_char']

        self._use_char_conv = config['use_char_conv']

        self._use_gaz = config['use_gaz']


        if config['use_gaz']:
            gazdir = config['GazetteerDir']
            gaz_names = config['Gazetteers']
            self._gazetteers = []
            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self._gazetteers.append(self._load_gaz_list(gazfile))

        # 游标
        self.cursor = 0

        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        all_samples = []

        # self.all_samples: [(tokens, labels),()]
        fea_len = len(self._vocabs)
        if self._use_gaz:
            fea_len += len(self._gazetteers)
        if self._use_char_conv:
            fea_len += self._max_char_len * 2



        for line in train_file:
            line = line.strip()
            if line:
                parts = line.split('|||')
                src_tokens = parts[0].strip().split(' ')
                tgt_tokens = parts[1].strip().split(' ')

                feaMat = numpy.zeros((len(src_tokens), fea_len),
                                     dtype='int32')

                label = numpy.zeros((len(tgt_tokens)), dtype='int32')
                for (lid, token) in enumerate(src_tokens):
                    parts = token.split('#')
                    for (i, voc) in enumerate(self._vocabs):
                        fpos = self._fea_pos[i]
                        wid = voc.getID(parts[fpos])
                        feaMat[lid, i] = wid
                    curr_end = len(self._vocabs)
                    if self._use_gaz:
                        gazStart = len(self._vocabs)
                        for (id, gaz) in enumerate(self._gazetteers):
                            if parts[0] in gaz:
                                feaMat[lid, id + gazStart] = 1
                        curr_end += len(self._gazetteers)
                    if self._use_char_conv:
                        word = parts[self._word_pos]
                        chStart = curr_end
                        chMaskStart = chStart + self._max_char_len
                        for i in range(len(word)):
                            if i >= self._max_char_len:
                                break
                            feaMat[lid, chStart + i] = self._vocab_char.getID(word[i])
                            feaMat[lid, chMaskStart + i] = 1
                num = len(tgt_tokens) - 1
                for (lid, token) in enumerate(tgt_tokens):
                    # if lid != num:
                    label[lid] = self._outtag_voc.getID(token)
                all_samples.append((feaMat, label))
        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def _load_gaz_list(self, file):
        words=set()
        with open(file) as f:
            for line in f:
                words.add(line.strip())
        return words


    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def next(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor+self.batch_size]
            # required_batch = self.all_samples[:config['batch_size']]
            self.cursor += self.batch_size
            input_seqs = [seq_label[0] for seq_label in required_batch]
            input_labels = [seq_label[1] for seq_label in required_batch]
            input_seqs_length = [s.shape[0] for s in input_seqs]
            input_labels_length = [s.size for s in input_labels]
            seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
            labels_padded = [pad_tgt_seq(s, max(input_labels_length))[numpy.newaxis, ...] for s in input_labels]
            # (batch, max_seq, len(embnames)+len(gazs)+max_char+max_char)
            seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)
            # (batch, max_label)
            label_tensor = torch.from_numpy(numpy.concatenate(labels_padded, axis=0)).type(torch.LongTensor)

            # input_seqs_length[-1] = 225
            # input_labels_length[-1] = 270

            return seq_tensor, label_tensor, input_labels_length, input_seqs_length
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)
        self.cursor = 0


class BioBatchGetter(object):
    def __init__(self, config, file_name, batch_size, shuffle=True, bio=False):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._vocabs = config['Vocabs']
        self._outtag_voc = config['BioOutTags']
        self._fea_pos = config['fea_pos']
        self._word_pos = config['WordPos']

        self._vocab_char = config['CharVoc']
        self._max_char_len = config['max_char']

        self._use_char_conv = config['use_char_conv']

        self._use_gaz = config['use_gaz']


        if config['use_gaz']:
            gazdir = config['GazetteerDir']
            gaz_names = config['Gazetteers']
            self._gazetteers = []
            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self._gazetteers.append(self._load_gaz_list(gazfile))

        # 游标
        self.cursor = 0

        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        all_samples = []

        # self.all_samples: [(tokens, labels),()]
        fea_len = len(self._vocabs)
        if self._use_gaz:
            fea_len += len(self._gazetteers)
        if self._use_char_conv:
            fea_len += self._max_char_len * 2



        for line in train_file:
            line = line.strip()
            if line:
                parts = line.split('|||')
                src_tokens = parts[0].strip().split(' ')
                tgt_tokens = parts[1].strip().split(' ')

                feaMat = numpy.zeros((len(src_tokens), fea_len),
                                     dtype='int32')

                label = numpy.zeros((len(tgt_tokens)-1), dtype='int32')
                for (lid, token) in enumerate(src_tokens):
                    parts = token.split('#')
                    for (i, voc) in enumerate(self._vocabs):
                        fpos = self._fea_pos[i]
                        wid = voc.getID(parts[fpos])
                        feaMat[lid, i] = wid
                    curr_end = len(self._vocabs)
                    if self._use_gaz:
                        gazStart = len(self._vocabs)
                        for (id, gaz) in enumerate(self._gazetteers):
                            if parts[0] in gaz:
                                feaMat[lid, id + gazStart] = 1
                        curr_end += len(self._gazetteers)
                    if self._use_char_conv:
                        word = parts[self._word_pos]
                        chStart = curr_end
                        chMaskStart = chStart + self._max_char_len
                        for i in range(len(word)):
                            if i >= self._max_char_len:
                                break
                            feaMat[lid, chStart + i] = self._vocab_char.getID(word[i])
                            feaMat[lid, chMaskStart + i] = 1
                num = len(tgt_tokens) - 1
                for (lid, token) in enumerate(tgt_tokens):
                    if lid != num:
                        label[lid] = self._outtag_voc.getID(token)
                all_samples.append((feaMat, label))
        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def _load_gaz_list(self, file):
        words=set()
        with codecs.open(file, mode='rb', encoding='utf-8') as f:
            for line in f:
                words.add(line.strip())
        return words


    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def next(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor+self.batch_size]
            # required_batch = self.all_samples[:config['batch_size']]
            self.cursor += self.batch_size
            # 按句子长度从大到小排列
            required_batch.sort(key=lambda x: x[0].shape[0], reverse=True)
            input_seqs = [seq_label[0] for seq_label in required_batch]
            input_labels = [seq_label[1] for seq_label in required_batch]
            input_seqs_length = [s.shape[0] for s in input_seqs]
            input_labels_length = [s.size for s in input_labels]
            seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
            labels_padded = [pad_tgt_seq(s, max(input_labels_length))[numpy.newaxis, ...] for s in input_labels]
            # (batch, max_seq, len(embnames)+len(gazs)+max_char+max_char)
            seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)
            # (batch, max_label)
            label_tensor = torch.from_numpy(numpy.concatenate(labels_padded, axis=0)).type(torch.LongTensor)

            # input_seqs_length[-1] = 225
            # input_labels_length[-1] = 270

            return seq_tensor, label_tensor, input_labels_length, input_seqs_length
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)
        self.cursor = 0

class CMNBioBatchGetter(object):
    def __init__(self, config, file_name, batch_size, shuffle=True, bio=False):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._vocabs = config['Vocabs']
        self._outtag_voc = config['BioOutTags']
        self._fea_pos = config['fea_pos']
        self._word_pos = config['WordPos']

        # self._vocab_char = config['CharVoc']
        self._vocab_word = config['WordId']
        self._max_char_len = config['max_char']

        self._use_char_conv = config['use_char_conv']

        self._use_gaz = config['use_gaz']

        if config['use_gaz']:
            gazdir = config['GazetteerDir']
            gaz_names = config['Gazetteers']
            self._gazetteers = []
            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self._gazetteers.append(self._load_gaz_list(gazfile))

        # 游标
        self.cursor = 0

        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        all_samples = []

        # self.all_samples: [(tokens, labels),()]
        fea_len = len(self._vocabs)
        if self._use_gaz:
            fea_len += len(self._gazetteers)
        fea_len += 1  # the word where the char is
        if self._use_char_conv:
            fea_len += self._max_char_len * 2

        for line in train_file:
            line = line.strip()
            if line:
                parts = line.split('|||')
                src_tokens = parts[0].strip().split(' ')
                tgt_tokens = parts[1].strip().split(' ')

                feaMat = numpy.zeros((len(src_tokens), fea_len),
                                     dtype='int32')

                label = numpy.zeros((len(tgt_tokens) - 1), dtype='int32')
                for (lid, token) in enumerate(src_tokens):
                    parts = token.split('#')
                    for (i, voc) in enumerate(self._vocabs):
                        if i == 0:
                            if parts[0] in ['0','1','2','3','4','5','6','7','8','9']:
                                parts[0] = '%NUM%'
                        fpos = self._fea_pos[i]
                        wid = voc.getID(parts[fpos])
                        feaMat[lid, i] = wid
                    curr_end = len(self._vocabs)
                    if self._use_gaz:
                        gazStart = len(self._vocabs)
                        for (id, gaz) in enumerate(self._gazetteers):
                            if parts[5] in gaz:
                                feaMat[lid, id + gazStart] = 1
                        curr_end += len(self._gazetteers)
                    feaMat[lid, curr_end] = self._vocab_word.getID(parts[5])
                    curr_end += 1
                    if self._use_char_conv:
                        word = parts[self._word_pos]
                        chStart = curr_end
                        chMaskStart = chStart + self._max_char_len
                        for i in range(len(word)):
                            if i >= self._max_char_len:
                                break
                            feaMat[lid, chStart + i] = self._vocab_char.getID(word[i])
                            feaMat[lid, chMaskStart + i] = 1
                num = len(tgt_tokens) - 1
                for (lid, token) in enumerate(tgt_tokens):
                    if lid != num:
                        label[lid] = self._outtag_voc.getID(token)
                all_samples.append((feaMat, label))
        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def _load_gaz_list(self, file):
        words = set()
        with codecs.open(file, mode='rb', encoding='utf-8') as f:
            for line in f:
                words.add(line.strip())
        return words

    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def next(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor + self.batch_size]
            # required_batch = self.all_samples[:config['batch_size']]
            self.cursor += self.batch_size
            # 按句子长度从大到小排列
            required_batch.sort(key=lambda x: x[0].shape[0], reverse=True)

            input_seqs = [seq_label[0] for seq_label in required_batch]
            input_labels = [seq_label[1] for seq_label in required_batch]
            input_seqs_length = [s.shape[0] for s in input_seqs]
            input_labels_length = [s.size for s in input_labels]
            seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
            labels_padded = [pad_tgt_seq(s, max(input_labels_length))[numpy.newaxis, ...] for s in input_labels]
            # (batch, max_seq, len(embnames)+len(gazs)+max_char+max_char)
            seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)
            # (batch, max_label)
            label_tensor = torch.from_numpy(numpy.concatenate(labels_padded, axis=0)).type(torch.LongTensor)

            # input_seqs_length[-1] = 225
            # input_labels_length[-1] = 270

            return seq_tensor, label_tensor, input_labels_length, input_seqs_length
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)
        self.cursor = 0




# class Batch_gen(object):
#     def __init__(self, file_name, batch_size):
#         self.batch_size = batch_size
#         self.file = codecs.open(file_name, 'rb')
#         self.pairs = []
#         for line in self.file:
#             if line.strip():
#                 str_pair = line.strip().split('\t')
#                 str_seq = str_pair[0].split(' ')
#                 seq = []
#                 for i in str_seq:
#                     seq.append(int(i))
#                 int_pair = [seq, int(str_pair[1])]
#                 self.pairs.append(int_pair)
#         self.cursor = 0
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         if self.cursor < len(self.pairs):
#             result = self.pairs[self.cursor:self.cursor+self.batch_size]
#             self.cursor += self.batch_size
#             seq_pad = []
#             label_pad
#             biggest = 0
#             for pair in result:
#                 length = len(pair[0])
#                 lengths.append(length)
#                 if length > biggest:
#                     biggest = length
#             for pair in result:
#                 if len(pair)
#
#             return result
#         else:
#             raise StopIteration("out of list")
# data = Batch_gen('data', 4)
# for i, batch in enumerate(data):
#     print i, batch
if __name__ == "__main__":
    config = get_conf('cmn')
    batch_getter = CMNBioBatchGetter(config, 'data/bio_cmn_dev.txt', 8, shuffle=False)
    for this_batch in batch_getter:
        pass
    # a = batch_getter.next()

    all = batch_getter.all_samples
    input_seqs = [seq_label[0] for seq_label in all]
    input_labels = [seq_label[1] for seq_label in all]
    input_seqs_length = [s.shape[0] for s in input_seqs]
    input_labels_length = [s.size for s in input_labels]
    print max(input_seqs_length), max(input_labels_length)
    pass
    # print get_source_mask(3, 4, [4,6,5]).transpose(0,1)

