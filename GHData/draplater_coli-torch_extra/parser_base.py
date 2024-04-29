import os
import pickle
import random
import shutil
import stat
import sys
import time

import torch
from abc import ABCMeta, abstractproperty, abstractmethod
from typing import List, Tuple, Generic, TypeVar, Type, Any, Optional

import numpy as np
from dataclasses import dataclass, field
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from coli.basic_tools.common_utils import T, NoPickle, AttrDict, Progbar, IdentityGetAttr, try_cache_keeper
from coli.basic_tools.dataclass_argparse import argfield, pretty_format, merge_predict_time_options
from coli.basic_tools.logger import log_to_file
from coli.parser_tools.magic_pack import read_script, write_script, get_codes
from coli.parser_tools.parser_base import DependencyParserBase, DF
from coli.data_utils.dataset import SentenceFeaturesBase, bucket_types, SentenceBucketsBase, HParamsBase, \
    DataFormatBase, SimpleSentenceBuckets, StreamingSentenceBuckets
from coli.torch_extra.bert_manager import BERTPlugin
from coli.torch_extra.dataset import ExternalEmbeddingPlugin
from coli.torch_extra.elmo_manager import ELMoPlugin
from coli.torch_extra.layers import AdvancedLearningOptions, ExternalContextualEmbedding, OptimizerOptions
from coli.torch_extra.utils import to_cuda
from tensorboardX import SummaryWriter

BK = TypeVar("BK", bound=SentenceBucketsBase)
SF = TypeVar("SF", bound=SentenceFeaturesBase)

OptionsType = TypeVar("OptionsType")
STOP_TRAINING = object()


class PyTorchParserBase(Generic[DF, SF], DependencyParserBase[DF], metaclass=ABCMeta):
    sentence_feature_class: Any = abstractproperty()

    statistics: Any
    network: Module

    def __init__(self, args: Any, data_train):
        super(PyTorchParserBase, self).__init__(args, data_train)
        self.global_step = 0
        self.global_epoch = 0
        self.codes = None

    # noinspection PyMethodOverriding
    def train(self, train_bucket: BK,
              dev_buckets: List[Tuple[str, List[DF], BK]]):
        pass

    def sentence_convert_func(self, sent_idx: int, sentence: T,
                              padded_length: int):
        return self.sentence_feature_class.from_sentence_obj(
            sent_idx, sentence, self.statistics,
            padded_length,
            plugins=self.plugins
        )

    def create_bucket(self, bucket_type: Type[BK], data: T, is_train) -> BK:
        return bucket_type(
            data, self.sentence_convert_func,
            self.hparams.train_batch_size if is_train else self.hparams.test_batch_size,
            self.hparams.num_buckets,
            self.sentence_feature_class,
            seed=self.hparams.seed,
            max_sentence_batch_size=self.hparams.max_sentence_batch_size
        )

    @abstractmethod
    def predict_bucket(self, test_buckets, **kwargs):
        raise NotImplementedError

    def predict(self, data_test, **kwargs):
        test_buckets = self.create_bucket(
            bucket_types[self.hparams.bucket_type], data_test, False)
        for i in self.predict_bucket(test_buckets, **kwargs):
            yield i

    @classmethod
    def repeat_train_and_validate(cls, data_train, data_devs, data_test, options):
        np.random.seed(options.hparams.seed)
        torch.manual_seed(options.hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.hparams.seed)
        parser = cls(options, data_train)
        log_to_file(parser.get_log_file(options))
        parser.logger.info('Options:\n%s', pretty_format(options))
        parser.logger.info('Network:\n%s', pretty_format(parser.network))

        train_buckets: BK = parser.create_bucket(
            bucket_types[options.hparams.bucket_type],
            data_train, True)
        dev_buckets: List[BK] = [
            parser.create_bucket(
                bucket_types[options.hparams.bucket_type], data_dev, False)
            for data_dev in data_devs.values()]

        # prevent source code change when training
        if parser.codes is None:
            parser.codes = get_codes(os.path.join(
                os.path.dirname(__file__), "../../"))

        while True:
            current_step = parser.global_step
            if current_step > options.hparams.train_iters:
                break
            ret = parser.train(
                train_buckets,
                [(a, b, c) for (a, b), c  # file_name, data, buckets
                 in zip(data_devs.items(), dev_buckets)]
            )
            if ret == STOP_TRAINING:
                break

    def post_load(self, new_options):
        self.options.__dict__.update(new_options.__dict__)

    @classmethod
    def load(cls, prefix, new_options: Optional[AttrDict] = None):
        if new_options is None:
            new_options = AttrDict()

        with open(prefix, "rb") as f:
            read_script(f)
            # noinspection PyUnusedLocal
            codes = pickle.load(f)
            entrance_class = pickle.load(f)
            if cls.__module__ != "__main__" and entrance_class is not cls:
                raise Exception(f"Not a model of {cls}. It is a {entrance_class}")
            self = torch.load(f, map_location="cpu" if not new_options.gpu else "cuda")
        self.post_load(new_options)
        return self

    def save(self, prefix, latest_filename=None):
        # prevent source code change when training
        if self.codes is None:
            self.codes = get_codes(os.path.join(
                os.path.dirname(__file__), "../../"))

        with open(prefix, "wb") as f:
            # noinspection PyTypeChecker
            write_script(f)
            pickle.dump(self.codes, f)
            pickle.dump(self.__class__, f)
            torch.save(self, f)
        st = os.stat(prefix)
        os.chmod(prefix, st.st_mode | stat.S_IEXEC)


class SimpleParser(Generic[OptionsType, DF, SF], PyTorchParserBase[DF, SF],
                   metaclass=ABCMeta):
    optimizer: Optimizer
    scheduler: ReduceLROnPlateau
    trainable_parameters: List[Parameter]

    @dataclass
    class HParams(HParamsBase):
        optimizer: OptimizerOptions = argfield(default_factory=OptimizerOptions)
        learning: AdvancedLearningOptions = field(
            default_factory=AdvancedLearningOptions)
        pretrained_contextual: ExternalContextualEmbedding.Options = field(
            default_factory=ExternalContextualEmbedding.Options)

    @dataclass
    class Options(PyTorchParserBase.Options):
        embed_file: Optional[str] = argfield(None, predict_time=True, predict_default=None)
        gpu: bool = argfield(False, predict_time=True, predict_default=False)
        hparams: Any = argfield(
            default_factory=lambda: SimpleParser.HParams(),
        )

    def __init__(self, args: OptionsType, data_train):
        super(SimpleParser, self).__init__(args, data_train)

        self.args: "SimpleParser.Options" = args
        self.hparams: "SimpleParser.HParams" = args.hparams
        self.plugins = {}

        pretrained_external_embedding = ExternalContextualEmbedding.get(self.hparams.pretrained_contextual)
        if pretrained_external_embedding is not None:
            self.plugins["pretrained_contextual"] = pretrained_external_embedding

        if args.embed_file is not None:
            @try_cache_keeper((args.embed_file, args.gpu))
            def get_external_embedding():
                return ExternalEmbeddingPlugin(args.embed_file, gpu=args.gpu)

            self.plugins["external_embedding"] = get_external_embedding()

        self.global_step = 0
        self.global_epoch = 1
        self.best_score = 0

        self.progbar = NoPickle(Progbar(self.hparams.train_iters, log_func=self.file_logger.info))
        self.writer: SummaryWriter = NoPickle(SummaryWriter(f"{self.args.output}/tensorboard-{self.logger_timestamp}/"))
        self.grad_clip_threshold = np.inf if self.hparams.learning.clip_grad_norm == 0 \
            else self.hparams.learning.clip_grad_norm

        assert self.hparams.print_every % self.hparams.learning.update_every == 0
        assert self.hparams.evaluate_every % self.hparams.print_every == 0

    def get_optimizer_and_scheduler(self, trainable_params, hparams=None):
        if hparams is None:
            hparams = self.hparams.optimizer
        optimizer = hparams.get(trainable_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max',
            factor=self.hparams.learning.step_decay_factor,
            patience=self.hparams.learning.step_decay_patience,
            verbose=True,
        )

        return optimizer, scheduler

    @property
    def file_logger(self):
        if getattr(self, "_file_logger", None) is None:
            self._file_logger = NoPickle(
                self.get_logger(self.options, False))
        return self._file_logger

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def schedule_lr(self, iteration):
        iteration = iteration + 1
        warmup_coeff = self.hparams.optimizer.learning_rate / \
                       self.hparams.learning.learning_rate_warmup_steps
        if iteration <= self.hparams.learning.learning_rate_warmup_steps:
            self.set_lr(iteration * warmup_coeff)

    def do_forward_and_backward(
            self, train_data, batch_sents, feed_dict):
        stack = [(batch_sents, feed_dict)]
        batch_devided = False
        total_count = 0
        failure_when_smallest_limit = 3
        need_gc = False
        need_empty_cache = False

        while stack:
            output = None

            if need_gc:
                import gc
                for i in range(3):
                    gc.collect()
                need_gc = False
            if need_empty_cache:
                for i in range(3):
                    torch.cuda.empty_cache()
                    time.sleep(3)
                need_empty_cache = False

            batch_sents, feed_dict = stack.pop()
            if self.args.gpu:
                to_cuda(feed_dict)
            try:
                output = self.network(batch_sents, AttrDict(feed_dict))
                # TODO: change bahaviour of update_every
                if self.hparams.learning.update_every > 1:
                    output.loss /= self.hparams.learning.update_every
                if batch_devided:
                    output.loss *= output.sent_count
                    total_count += int(output.sent_count)
                output.loss.backward()
                output.loss = output.loss.detach()
                yield output
                failure_when_smallest_limit = 3
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise
                else:
                    batch_devided = True
                    self.logger.info(e)
                    self.logger.info(f"Out of memory with {len(batch_sents)} sentence, "
                                     f"Max sentence length is {max(len(i.original_obj) for i in batch_sents)}. "
                                     f"Clear cache, split batch and try again automatically."
                                     f"If this appear frequently, you should use a smaller batch size.")
                    del output
                    if len(batch_sents) == 1:
                        self.logger.info("Even the smallest batch cause OOM.")
                        # may be we still have chance yet?
                        failure_when_smallest_limit -= 1
                        if failure_when_smallest_limit > 0:
                            # some variable is reference by the exception stack
                            # need to clear cache out of here
                            need_gc = True
                            need_empty_cache = True
                            stack.append((batch_sents, feed_dict))
                        else:
                            raise
                    else:
                        del feed_dict
                        need_empty_cache = True
                        split_point = int(len(batch_sents) / 2)
                        batch_sents_1 = batch_sents[:split_point]
                        if batch_sents_1:
                            # TODO: a better interface for this purpose
                            feed_dict_1 = train_data.return_batches(
                                batch_sents_1, pls=IdentityGetAttr(), batch_size=0, sort_key_func=None,
                                original=False, use_sub_batch=False, plugins=self.plugins)
                            stack.append((batch_sents_1, feed_dict_1))
                            del batch_sents_1, feed_dict_1
                        batch_sents_2 = batch_sents[split_point:]
                        if batch_sents_2:
                            # TODO: a better interface for this purpose
                            feed_dict_2 = train_data.return_batches(
                                batch_sents_2, pls=IdentityGetAttr(), batch_size=0, sort_key_func=None,
                                original=False, use_sub_batch=False, plugins=self.plugins)
                            stack.append((batch_sents_2, feed_dict_2))
                            del batch_sents_2, feed_dict_2
                del e

        if batch_devided:
            for p in self.network.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad /= total_count

    def train(self, train_data, dev_args_list=None):
        total_loss = 0.0
        sent_count = 0
        start_time = time.time()
        self.network.train()

        for batch_sents, feed_dict in train_data.generate_batches(
                self.hparams.train_batch_size,
                shuffle=True, original=True, plugins=self.plugins
                # sort_key_func=lambda x: x.sent_length
        ):
            self.schedule_lr(self.global_step)
            for output in self.do_forward_and_backward(
                    train_data, batch_sents, feed_dict):
                total_loss += output.loss.detach() * output.sent_count
                sent_count += output.sent_count
                del output
            if self.global_step % self.hparams.learning.update_every != 0:
                self.global_step += 1
                continue
            # noinspection PyArgumentList
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.global_step != 0 and \
                    self.global_step % self.hparams.print_every == 0:
                end_time = time.time()
                speed = sent_count / (end_time - start_time)
                start_time = end_time
                loss = total_loss / sent_count
                self.progbar.update(
                    self.global_step,
                    exact=[("Loss", loss), ("Speed", speed),
                           ("Epoch", self.global_epoch)]
                )
                self.writer.add_scalar("loss", loss, global_step=self.global_step)
                sent_count = 0
                total_loss = 0.0
                if self.global_step % self.hparams.evaluate_every == 0:
                    if dev_args_list is not None:
                        for filename, data, buckets in dev_args_list:
                            self.evaluate_when_training(buckets, data, filename)
                    self.network.train()
            self.global_step += 1

        self.global_epoch += 1
        self.progbar.finish()
        if self.global_step > self.hparams.learning.learning_rate_warmup_steps:
            self.scheduler.step(self.best_score)
            for i, param_group in enumerate(self.optimizer.param_groups):
                if param_group['lr'] < self.hparams.learning.min_learning_rate:
                    return STOP_TRAINING

    def evaluate_when_training(self, buckets, data, filename):
        output_file = self.get_output_name(
            self.args.output, filename, self.global_step)
        best_output_file = self.get_output_name(
            self.args.output, filename, "best")
        sys.stdout.write("\n")
        outputs = list(self.predict_bucket(buckets))
        if hasattr(outputs[0], "to_string"):
            for sample_idx, sample in enumerate(random.sample(outputs, 3)):
                self.writer.add_text("sample_outputs", sample.to_string(), self.global_step + sample_idx)
        self.write_result(output_file, outputs)
        self.evaluate_and_update_best_score(
            data, filename, outputs, output_file, best_output_file,
            log_file=output_file + ".txt")

    @abstractmethod
    def split_batch_result(self, batch_result):
        raise NotImplementedError

    @abstractmethod
    def merge_answer(self, sent, answer):
        raise NotImplementedError

    def get_parsed(self, bucket):
        self.network.eval()
        with torch.no_grad():
            for batch_sent, feed_dict in bucket.generate_batches(
                    self.hparams.test_batch_size, original=True,
                    plugins=self.plugins
            ):
                if self.args.gpu:
                    to_cuda(feed_dict)
                results = self.network(batch_sent, AttrDict(feed_dict))
                for original, parsed in zip(batch_sent, self.split_batch_result(results)):
                    yield original, parsed

    def evaluate(self, gold, gold_file, outputs, output_file, log_file):
        data_format_class = self.get_data_formats()[self.options.data_format]
        if getattr(data_format_class, "has_internal_evaluate_func", False):
            return data_format_class.internal_evaluate(
                gold, outputs, log_file, print=False)
        else:
            return data_format_class.evaluate_with_external_program(
                gold_file, output_file, perf_file=log_file, print=False)

    def evaluate_and_update_best_score(self, gold, gold_file_name, outputs,
                                       output_file, best_output_file, log_file=None):
        last_best_score = self.best_score
        score = self.evaluate(gold, gold_file_name, outputs, output_file, log_file)
        self.writer.add_scalar("score", score, global_step=self.global_step)
        if score > last_best_score:
            self.logger.info("New best score: {:.2f} > {:.2f}, saving model...".format(
                score, last_best_score))
            self.best_score = score
            self.save(self.args.output + "/model")
            shutil.copyfile(output_file, best_output_file)
            shutil.copyfile(log_file, best_output_file + ".txt")
        else:
            self.logger.info("No best score: {:.2f} <= {:.2f}".format(score, last_best_score))
        return score

    # noinspection PyMethodOverriding
    def predict_bucket(self, bucket):
        if isinstance(bucket, SimpleSentenceBuckets) or isinstance(bucket, StreamingSentenceBuckets):
            for sent_feature, answer in self.get_parsed(bucket):
                yield self.merge_answer(sent_feature, answer)
        else:
            outputs: List[Any] = []
            for sent_feature, answer in self.get_parsed(bucket):
                if len(outputs) < sent_feature.original_idx + 1:
                    outputs.extend([None] * (sent_feature.original_idx + 1 - len(outputs)))
                outputs[sent_feature.original_idx] = self.merge_answer(sent_feature, answer)
            yield from outputs

    def post_load(self, new_options):
        self.logger.info(f"New options: {pretty_format(new_options, is_training=False)}")
        merge_predict_time_options(self.options, new_options)
        self.logger.info(f"Merged Options: {pretty_format(self.options, is_training=False)}")

        if "external_embedding" in self.plugins:
            assert new_options.embed_file, "Embedding file is required"
            self.plugins["external_embedding"].reload(new_options.embed_file, gpu=new_options.gpu)

        if "pretrained_contextual" in self.plugins:
            plugin = self.plugins["pretrained_contextual"]
            if isinstance(plugin, ELMoPlugin):
                plugin.reload(self.hparams.pretrained_contextual.elmo_options.path,
                              self.options.gpu)
            elif isinstance(plugin, BERTPlugin):
                plugin.reload(self.hparams.pretrained_contextual.bert_options.bert_model,
                              self.options.gpu)
            else:
                raise NotImplementedError(f"{plugin.__class__}")
