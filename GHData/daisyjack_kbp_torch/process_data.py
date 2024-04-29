# coding: utf-8

import torch
import numpy
import os
import codecs

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

class AnnotationBatchGetter(object):
    def __init__(self, config, batch_size):
        # self.encoding = config['encoding']
        self.cursor = 0
        self.all_samples = []
        self.sample_num = len(self.all_samples)
        self.config = config
        self.batch_size = batch_size
        self._vocabs = config['Vocabs']
        self._outtag_voc = config['BioOutTags']  # self._outtag_voc = config['OutTags']
        # self._tag_pos = config['ner_pos']
        self._word_pos = config['WordPos']
        self._fea_pos = config['fea_pos']

        self._vocab_char = config['CharVoc']
        self._vocab_word = config['WordId']
        self._max_char_len = config['max_char']

        self._use_char_conv = config['use_char_conv']

        self._use_gaz = config['use_gaz']
        if self._use_gaz:
            gazdir = config['GazetteerDir']
            gaz_names = config['Gazetteers']
            self._gazetteers = []
            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self._gazetteers.append(self._load_gaz_list(gazfile))

    def _load_gaz_list(self, file):
        words = set()
        with codecs.open(file, mode='rb', encoding='utf-8') as f:
            for line in f:
                words.add(line.strip())
        return words

    def get_feature(self, tokens):
        fea_len = len(self._vocabs)
        if self._use_gaz:
            fea_len += len(self._gazetteers)
        if self._use_char_conv:
            fea_len += self._max_char_len * 2
        if self.config['lang'] == 'cmn':
            fea_len += 1
        feaMat = numpy.zeros((len(tokens), fea_len),
                             dtype='int32')
        for (lid, token) in enumerate(tokens):
            parts = [token['word_lower'], token['word'],
                     token['caps'], token['pos'],
                     token['ner']]
            if token.has_key('comb-word'):
                parts.append(token['comb-word'])  # .encode('utf-8'))
            for (i, voc) in enumerate(self._vocabs):
                fpos = self._fea_pos[i]
                wid = voc.getID(parts[fpos])
                feaMat[lid, i] = wid
            curr_end = len(self._vocabs)
            if self._use_gaz:
                gazStart = len(self._vocabs)
                for (id, gaz) in enumerate(self._gazetteers):
                    if self.config['lang'] == 'cmn':
                        if parts[5] in gaz:
                            feaMat[lid, id + gazStart] = 1
                    else:
                        if parts[0] in gaz:
                            feaMat[lid, id + gazStart] = 1
                curr_end += len(self._gazetteers)
            if self.config['lang'] == 'cmn':
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

        return feaMat

    def use_annotaion(self, text_spans):
        all_samples = []
        for id, sentence in enumerate(text_spans):
            feaMat = self.get_feature(sentence['tokens'])
            all_samples.append(feaMat)
        self.cursor = 0
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        input_seqs = self.all_samples
        input_seqs_length = [s.shape[0] for s in input_seqs]
        seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
        seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)

        return seq_tensor, 0, [0], input_seqs_length

    def next(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor + self.batch_size]
            # required_batch = self.all_samples[:config['batch_size']]
            self.cursor += self.batch_size
            # # 按句子长度从大到小排列
            # required_batch.sort(key=lambda x: x[0].shape[0], reverse=True)

            input_seqs = required_batch
            # input_labels = [seq_label[1] for seq_label in required_batch]
            input_seqs_length = [s.shape[0] for s in input_seqs]
            # input_labels_length = [s.size for s in input_labels]
            seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
            # labels_padded = [pad_tgt_seq(s, max(input_labels_length))[numpy.newaxis, ...] for s in input_labels]
            # (batch, max_seq, len(embnames)+len(gazs)+max_char+max_char)
            seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)
            # (batch, max_label)
            # label_tensor = torch.from_numpy(numpy.concatenate(labels_padded, axis=0)).type(torch.LongTensor)

            # input_seqs_length[-1] = 225
            # input_labels_length[-1] = 270

            return seq_tensor, 0, [0], input_seqs_length
        else:
            raise StopIteration("out of list")

