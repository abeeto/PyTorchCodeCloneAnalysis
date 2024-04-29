from typing import Optional

import torch

from dataclasses import dataclass
from torch.nn import Linear

from coli.basic_tools.common_utils import NoPickle, AttrDict
from coli.basic_tools.dataclass_argparse import argfield, OptionsBase, ExistFile
from coli.basic_tools.logger import logger
from coli.data_utils.dataset import SentenceFeaturesBase, START_OF_SENTENCE, END_OF_SENTENCE
from coli.torch_extra.dataset import InputPluginBase
from coli.torch_extra.utils import pad_and_stack_1d, broadcast_gather
from coli.torch_extra.dropout import FeatureDropout2

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
}


class BERTPlugin(InputPluginBase):
    @dataclass
    class Options(OptionsBase):
        bert_model: ExistFile = argfield(predict_time=True)
        student_model: Optional[str] = argfield(default=None, predict_time=True)
        lower: bool = False
        project_to: Optional[int] = None
        feature_dropout: float = 0.0
        finetune_last_n: int = 0
        pooling_method: Optional[str] = "last"

    def __init__(self, bert_model, student_model=None, lower=False, project_to=None,
                 feature_dropout=0.0,
                 pooling_method="last",
                 gpu=False, finetune_last_n=0):
        super().__init__()
        self.lower = lower
        self.finetune_tune_last_n = finetune_last_n
        self.pooling_method = pooling_method

        self.reload(bert_model, gpu, student_model)

        if feature_dropout > 0:
            self.feature_dropout_layer = FeatureDropout2(feature_dropout)

        if project_to:
            self.projection = Linear(self.output_dim, project_to, bias=False)
            self.output_dim = project_to
        else:
            self.projection = None

    def reload(self, bert_model, gpu, student_model=None):
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        logger.info(f"Reload BERT from {bert_model}")
        if bert_model.endswith('.tar.gz'):
            self.tokenizer = NoPickle(BertTokenizer.from_pretrained(
                bert_model.replace('.tar.gz', '-vocab.txt'),
                do_lower_case=self.lower))
        else:
            self.tokenizer = NoPickle(
                BertTokenizer.from_pretrained(bert_model, do_lower_case=self.lower))

        self.bert = NoPickle(BertModel.from_pretrained(bert_model))
        if gpu:
            self.bert = NoPickle(self.bert.cuda())
        self.output_dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings

        for p in self.bert.parameters():
            p.requires_grad = False

        if self.finetune_tune_last_n > 0:
            self.finetune_layers = self.bert.encoder.layer[-self.finetune_tune_last_n:]
            for p in self.finetune_layers.parameters():
                p.requires_grad = True

        if student_model is not None:
            self.word_embeddings = self.bert.embeddings.word_embeddings
            from local_scripts.bert_distillation import BertDistillator
            self.student_model = BertDistillator.load(
                student_model, AttrDict(gpu=gpu)).network
            self.word_embeddings = NoPickle(self.word_embeddings)
            self.student_model = NoPickle(self.student_model)
            for p in self.student_model:
                p.requires_grad = False
            self.bert = None
        else:
            self.student_model = None

    def process_sentence_feature(self, sent, sent_feature: SentenceFeaturesBase,
                                 padded_length, start_and_end=False):
        keep_boundaries = start_and_end
        word_pieces = ["[CLS]"]
        word_starts = []
        word_ends = []
        if keep_boundaries:
            word_starts.append(0)
            word_ends.append(0)

        for maybe_words in sent.words:
            word_starts.append(len(word_pieces))
            if maybe_words == START_OF_SENTENCE or maybe_words == END_OF_SENTENCE:
                continue
            for word in maybe_words.split("_"):  # like more_than in deepbank:
                word = BERT_TOKEN_MAPPING.get(word, word)
                pieces = self.tokenizer.tokenize(word)
                word_pieces.extend(pieces)
            word_ends.append(len(word_pieces) - 1)

        word_pieces.append("[SEP]")
        if keep_boundaries:
            word_starts.append(len(word_pieces) - 1)
            word_ends.append(len(word_pieces) - 1)

        word_length = padded_length + (2 if keep_boundaries else 0)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_tokens"] = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_pieces))
        sent_feature.extra["bert_word_starts"] = torch.zeros((word_length,), dtype=torch.int64)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_word_starts"][:len(word_starts)] = torch.tensor(word_starts)
        sent_feature.extra["bert_word_ends"] = torch.zeros((word_length,), dtype=torch.int64)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_word_ends"][:len(word_ends)] = torch.tensor(word_ends)

    def process_batch(self, pls, feed_dict, batch_sentences):
        feed_dict["bert_tokens"] = pad_and_stack_1d(
            [i.extra["bert_tokens"] for i in batch_sentences])
        feed_dict["bert_word_ends"] = pad_and_stack_1d(
            [i.extra["bert_word_ends"] for i in batch_sentences])

    def forward(self, feed_dict):
        pooling_method = self.pooling_method
        all_input_mask = feed_dict.bert_tokens.gt(0)
        if self.student_model is None:
            all_encoder_layers, _ = self.bert(feed_dict.bert_tokens,
                                              attention_mask=all_input_mask)
            features = all_encoder_layers[-1]
        else:
            # use student model instead of original BERT
            word_embeddings = self.word_embeddings(
                feed_dict.bert_tokens)
            sent_lengths = all_input_mask.int().sum(-1)
            features = self.student_model.projection(self.student_model.lstm(
                word_embeddings, sent_lengths))
        if pooling_method == "last":
            word_features = broadcast_gather(features, 1, feed_dict.bert_word_ends)
        elif pooling_method == "none":
            word_features = features
        else:
            raise Exception(f"Invalid pooling method {pooling_method}")

        if self.projection is not None:
            word_features = self.projection(word_features)

        if hasattr(self, "feature_dropout_layer"):
            word_features = self.feature_dropout_layer(word_features)

        # noinspection PyUnboundLocalVariable
        return word_features
