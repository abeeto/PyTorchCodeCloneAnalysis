import datasets
import os
import logging
from numpy import True_
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,squad_convert_examples_to_features)
from tokenization_kobert import KoBertTokenizer
from transformers.data.processors.squad import SquadV2Processor,SquadV1Processor
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from torch.utils.data import random_split
logger = logging.getLogger(__name__)


class KorQuadDataModule(LightningDataModule):
    def __init__(
        self,
        hparams,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model_name_or_path = self.hparams.model_name_or_path
        self.output_dir = self.hparams.data_output_dir
        self.data_dir = self.hparams.data_dir
        self.max_seq_length = self.hparams.max_seq_length
        self.doc_stride = self.hparams.doc_stride
        self.max_query_length = self.hparams.max_query_length
        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size        
        version_2_with_negative =True
        
        self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert",do_lower_case=True)
        
    def setup(self,stage=None):
        self.dataset = {}
        train_dataset = self.load_and_cache_examples(self.tokenizer, evaluate=False)
        train_len = int(len(train_dataset) * 0.8)
        val_len = len(train_dataset) - train_len
        self.dataset["train"], self.dataset["validation"] = random_split(train_dataset, [train_len,val_len])
        self.dataset["test"], self.dataset["features"],  self.dataset["examples"] = self.load_and_cache_examples(self.tokenizer, evaluate=True, output_examples = True)

    def load_and_cache_examples(self, tokenizer, evaluate=False, output_examples=False):
        # Load data features from cache or dataset file
        input_dir = "data/"
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq_length),
            ),
        )

        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"]
            )
        else:
            logger.info("Creating features from dataset file at %s", input_dir)

            processor = SquadV1Processor()
            if evaluate:
                dev_file = "KorQuAD_v1.0_dev.json"
                examples = processor.get_dev_examples(self.data_dir, filename=dev_file)
            else:
                train_file = "KorQuAD_v1.0_train.json"
                examples = processor.get_train_examples(self.data_dir, filename=train_file)

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
            )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples" : examples}, cached_features_file)
            
        if output_examples:
            return dataset, features, examples
        return dataset

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.train_batch_size, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, pin_memory=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.train_batch_size, pin_memory=True, num_workers=4)


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        hparams,
        **kwargs,
    ):
        super().__init__()
        
        self.hparams = hparams
        
        #self.model_name_or_path = self.hparams.model_name_or_path
        self.model_name_or_path  = "albert-base-v2"
        self.task_name = self.hparams.task_name
        self.max_seq_length = self.hparams.max_seq_length
        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

        self.text_fields = self.task_text_field_map[self.task_name]
        self.num_labels = self.glue_task_num_labels[self.task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features