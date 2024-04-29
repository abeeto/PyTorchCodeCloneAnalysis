from Config import config
from ReadCorpus import eng_sentences, fra_sentences, eng_dict, fra_dict
import numpy
import torch


class DataLoader:
    def __init__(self):
        self.eng_word_idx = []
        self.fra_word_idx = []
        for sent1, sent2 in zip(eng_sentences, fra_sentences):
            self.eng_word_idx.append(self._to_idx(sent1, eng_dict))
            self.fra_word_idx.append(self._to_idx(sent2, fra_dict))

    @staticmethod
    def _to_idx(sent, dic):
        assert len(sent) <= 10
        word_idx = []
        for word in sent:
            if word not in dic:
                word_idx.append(0)
            else:
                word_idx.append(dic[word])
        while len(word_idx) < 10:
            word_idx.append(dic['[PAD]'])
        return word_idx

    def _to_tensor(self, begin, end):
        eng_tensor = torch.from_numpy(numpy.array(self.eng_word_idx[begin:end])).to(config.device)
        fra_tensor = torch.from_numpy(numpy.array(self.fra_word_idx[begin:end])).to(config.device)
        return eng_tensor, fra_tensor

    def run(self):
        for i in range(len(self.fra_word_idx) // config.batch_size):
            begin = i * config.batch_size
            end = (i + 1) * config.batch_size
            yield self._to_tensor(begin, end)


loader = DataLoader()
