import os

import torch
from typing import List, Optional
from weakref import WeakValueDictionary

from dataclasses import dataclass
from torch.nn import Module, Linear

from coli.basic_tools.common_utils import NoPickle
from coli.basic_tools.dataclass_argparse import argfield, OptionsBase
from coli.data_utils.dataset import SentenceFeaturesBase
from coli.torch_extra.dataset import InputPluginBase
from coli.torch_extra.utils import pad_and_stack_1d
from coli.torch_extra.dropout import FeatureDropout2

global_elmo_cache = WeakValueDictionary()


def get_elmo(options_file: str,
             weight_file: str,
             num_output_representations: int,
             requires_grad: bool = False,
             do_layer_norm: bool = False,
             dropout: float = 0.5,
             vocab_to_cache: List[str] = None,
             keep_sentence_boundaries: bool = False,
             ):
    from allennlp.modules import Elmo
    key = (options_file, weight_file)
    old_elmo = global_elmo_cache.get(key)
    if old_elmo:
        # noinspection PyProtectedMember
        module = old_elmo._elmo_lstm
        options_file = None
        weight_file = None
    else:
        module = None

    ret = Elmo(options_file=options_file,
               weight_file=weight_file,
               num_output_representations=num_output_representations,
               requires_grad=requires_grad,
               do_layer_norm=do_layer_norm,
               dropout=dropout,
               vocab_to_cache=vocab_to_cache,
               keep_sentence_boundaries=keep_sentence_boundaries,
               module=module)

    if not old_elmo:
        global_elmo_cache[key] = ret

    return ret


class ELMoPlugin(InputPluginBase):
    @dataclass
    class Options(OptionsBase):
        path: str = argfield(predict_time=True)
        requires_grad: bool = False
        do_layer_norm: bool = False
        dropout: float = 0.5
        feature_dropout: float = 0.0
        keep_sentence_boundaries: bool = False
        project_to: Optional[int] = None

    def __init__(self, path: str,
                 num_output_representations: int = 1,
                 requires_grad: bool = False,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 feature_dropout: float = 0.1,
                 vocab_to_cache: List[str] = None,
                 keep_sentence_boundaries: bool = False,
                 project_to: Optional[int] = None,
                 gpu=False
                 ):
        from bilm.load_vocab import BiLMVocabLoader
        super(ELMoPlugin, self).__init__()
        assert num_output_representations == 1
        self.num_output_representations = 1
        self.requires_grad = requires_grad
        self.do_layer_norm = do_layer_norm
        self.dropout = dropout
        self.vocab_to_cache = vocab_to_cache
        self.keep_sentence_boundaries = keep_sentence_boundaries
        self.vocab = BiLMVocabLoader(path)
        self.project_to = project_to
        self.reload(path, gpu)

        self.feature_dropout_layer = FeatureDropout2(feature_dropout)

        if self.project_to:
            self.projection = Linear(self.output_dim, project_to, bias=False)
            self.output_dim = self.project_to

    def reload(self, bilm_path, gpu):
        self.elmo = get_elmo(os.path.join(bilm_path, 'options.json'),
                             os.path.join(bilm_path, 'lm_weights.hdf5'),
                             self.num_output_representations,
                             self.requires_grad, self.do_layer_norm,
                             self.dropout, self.vocab_to_cache,
                             self.keep_sentence_boundaries
                             )

        if gpu:
            self.elmo.cuda()

        if not self.requires_grad:
            self.elmo = NoPickle(self.elmo)
            if not getattr(self, "scalar_mix", None):
                self.scalar_mix = self.elmo.scalar_mix_0
            else:
                self.elmo.scalar_mix_0 = self.scalar_mix

        self.output_dim = 1024

        if self.project_to:
            self.elmo.scalar_mix_0.gamma.requires_grad = False

    def process_sentence_feature(self, sent, sent_feature: SentenceFeaturesBase,
                                 padded_length, start_and_end=False):
        bilm_chars_padded = torch.from_numpy(self.vocab.get_chars_input(
            sent.words, padded_length))
        sent_feature.extra["bilm_characters"] = bilm_chars_padded

    def process_batch(self, pls, feed_dict, batch_sentences):
        feed_dict[pls.bilm_chars] = pad_and_stack_1d([i.extra["bilm_characters"]
                                                      for i in batch_sentences])

    def forward(self, feed_dict):
        ret = self.elmo(feed_dict.bilm_chars)["elmo_representations"][0]
        if self.project_to:
            ret = self.projection(ret)
        if hasattr(self, "feature_dropout_layer"):
            ret = self.feature_dropout_layer(ret)
        return ret
