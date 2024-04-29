from abc import ABCMeta
from distutils.version import StrictVersion
from typing import List, Dict

import torch
from torch.nn import Module, Embedding, Linear

from coli.basic_tools.common_utils import NoPickle
from coli.data_utils import dataset
from coli.data_utils.embedding import read_embedding
from coli.torch_extra.utils import pad_and_stack_1d

try:
    torch_version = StrictVersion(torch.__version__)
except ValueError:
    torch_version = StrictVersion("1.1.0")

if torch_version >= StrictVersion("1.1.0"):
    @torch.jit.script
    class TorchDictionary(object):
        def __init__(self, dict_input: Dict[str, int]):
            self.dict = dict_input

        def lookup(self, tokens: List[str], padded_length: int,
                   default: int, dtype: int = torch.int64,
                   start_and_stop: bool = False):
            ret = torch.zeros([padded_length + (2 if start_and_stop else 0)],
                              dtype=dtype)
            if start_and_stop:
                ret[0] = self.dict.get("___START___", default)
            idx = 1 if start_and_stop else 0
            for idx in range(len(tokens)):
                ret[idx] = self.dict.get(tokens[idx], default)
            if start_and_stop:
                ret[idx + 1] = self.dict.get("___END___", default)

            return ret
else:
    class TorchDictionary(object):
        def __init__(self, dict_input: Dict[str, int]):
            raise Exception("requires pytorch > 1.1.0")


def lookup_list(tokens_itr, token_dict, padded_length,
                default, dtype=torch.int64,
                start_and_stop=False,
                tensor_factory=torch.zeros):
    return dataset.lookup_list(tokens_itr, token_dict, padded_length,
                               default, dtype, start_and_stop, tensor_factory)


def lookup_characters(words_itr, char_dict, padded_length,
                      default, max_word_length=20, dtype=torch.int64,
                      start_and_stop=True,
                      sentence_start_and_stop=False,
                      return_lengths=False,
                      tensor_factory=torch.zeros
                      ):
    return dataset.lookup_characters(words_itr, char_dict, padded_length,
                                     default, max_word_length, dtype,
                                     start_and_stop, sentence_start_and_stop,
                                     return_lengths, tensor_factory)


class InputPluginBase(Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(InputPluginBase, self).__init__()

    def reload(self, *args, **kwargs):
        pass

    def process_sentence_feature(self, sent, sent_feature,
                                 padded_length, start_and_end=False):
        pass

    def process_batch(self, pls, feed_dict, batch_sentences):
        pass

    def forward(self, feed_dict):
        pass


class ExternalEmbeddingPlugin(InputPluginBase):
    def __init__(self, embedding_filename, project_to=None, encoding="utf-8",
                 lower=False, gpu=False):
        super().__init__()
        self.lower = lower
        self.project_to = project_to
        self.reload(embedding_filename, encoding, gpu)

    def reload(self, embedding_filename, encoding="utf-8", gpu=False):
        self.gpu = gpu
        words_and_vectors = read_embedding(embedding_filename, encoding)
        self.output_dim = len(words_and_vectors[0][1])
        # noinspection PyCallingNonCallable
        words_and_vectors.insert(0, ("*UNK*", [0.0] * self.output_dim))

        words, vectors_py = zip(*words_and_vectors)
        self.lookup = {word: idx for idx, word in enumerate(words)}
        # noinspection PyCallingNonCallable
        vectors = torch.tensor(vectors_py, dtype=torch.float32)

        # prevent .cuda()
        # noinspection PyReturnFromInit
        self.embedding_ = [NoPickle(Embedding.from_pretrained(vectors, freeze=True))]

        if self.project_to:
            self.projection = Linear(self.output_dim, self.project_to)
            self.output_dim = self.project_to

    def lower_func(self, word):
        if self.lower:
            return word.lower()
        return word

    def process_sentence_feature(self, sent, sent_feature,
                                 padded_length, start_and_end=False):
        words = [self.lower_func(i) for i in sent.words]
        sent_feature.extra["words_pretrained"] = lookup_list(
            words, self.lookup,
            padded_length=padded_length, default=0, start_and_stop=start_and_end)

    def process_batch(self, pls, feed_dict, batch_sentences):
        feed_dict[pls.words_pretrained] = pad_and_stack_1d([i.extra["words_pretrained"] for i in batch_sentences])

    def forward(self, feed_dict):
        ret = self.embedding_[0](feed_dict.words_pretrained.cpu())
        if self.gpu:
            ret = ret.cuda()
        if hasattr(self, "projection"):
            ret = self.projection(ret)
        return ret
