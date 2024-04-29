from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import datasets
from datasets import ClassLabel
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics
)

class Transformer(pl.LightningModule):

    class DataModule(pl.LightningDataModule):

        task_text_field_map = {
            'cola': ['sentence'],
            'sst2': ['sentence'],
            'mrpc': ['sentence1', 'sentence2'],
            'qqp': ['question1', 'question2'],
            'stsb': ['sentence1', 'sentence2'],
            'mnli': ['premise', 'hypothesis'],
            'qnli': ['question', 'sentence'],
            'rte': ['sentence1', 'sentence2'],
            'wnli': ['sentence1', 'sentence2'],
            'ax': ['premise', 'hypothesis']
        }

        loader_columns = [
            'datasets_idx',
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'start_positions',
            'end_positions',
            'labels'
        ]

        def __init__(self, classifier_instance):
            super().__init__()
            self.hparams = classifier_instance.hparams
            self.classifier = classifier_instance
            self.max_seq_length = 128

            if self.hparams.train_data:
                self.text_fields = self.hparams.text_fields.split()
            else:
                self.text_fields = self.task_text_field_map[self.hparams.task_name]
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, use_fast=True)

        def setup(self, stage):

            if self.hparams.train_data:
                self.dataset = datasets.load_dataset('csv', data_files={'train': self.hparams.train_data,
                                                                        'validation': self.hparams.dev_data})
                new_features = self.dataset['train'].features.copy()
                new_features["label"] = ClassLabel(names=self.hparams.class_names.split())
                self.dataset['train'] = self.dataset['train'].cast(new_features)
            else:
                self.dataset = datasets.load_dataset('glue', self.hparams.task_name)

            if self.hparams.task_name == 'stsb':
                self.num_labels = 1
            else:
                self.num_labels = self.dataset['train'].features['label'].num_classes
                self.class_names = self.dataset['train'].features['label'].names

            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=['label'],
                )
                self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
                self.dataset[split].set_format(type="torch", columns=self.columns)

            self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]


        def prepare_data(self):
            if not self.hparams.train_data:
                datasets.load_dataset('glue', self.hparams.task_name)
            AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, use_fast=True)

        def train_dataloader(self):
            """ Function that loads the train set. """
            return DataLoader(self.dataset['train'], batch_size=self.hparams.batch_size)

        def val_dataloader(self):
            """ Function that loads the validation set. """
            if len(self.eval_splits) == 1:
                return DataLoader(self.dataset['validation'], batch_size=self.hparams.batch_size)
            elif len(self.eval_splits) > 1:
                return [DataLoader(self.dataset[x], batch_size=self.hparams.batch_size) for x in self.eval_splits]

        def test_dataloader(self):
            """ Function that loads the test set. """
            if len(self.eval_splits) == 1:
                return DataLoader(self.dataset['test'], batch_size=self.hparams.batch_size)
            elif len(self.eval_splits) > 1:
                return [DataLoader(self.dataset[x], batch_size=self.hparams.batch_size) for x in self.eval_splits]

        def convert_to_features(self, example_batch, indices=None):

            # Either encode single sentence or sentence pairs
            if len(self.text_fields) > 1:
                texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
            else:
                texts_or_text_pairs = example_batch[self.text_fields[0]]

            # Tokenize the text/text pairs
            features = self.tokenizer.batch_encode_plus(
                texts_or_text_pairs,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                truncation=True
            )

            # Rename label to labels to make it easier to pass to model forward
            features['labels'] = example_batch['label']

            return features

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        # Build Data module
        self.data = self.DataModule(self)
        self.data.prepare_data()
        self.data.setup('fit')

        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path, num_labels=self.data.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name_or_path, config=self.config)
        self.train_data = self.hparams.train_data
        self.dev_data = self.hparams.dev_data
        self.test_data = self.hparams.test_data
        self.eval_splits = self.data.eval_splits

        if self.train_data:
            if self.hparams.metric=='matthews_correlation':
                self.metric = datasets.load_metric('glue', 'cola')
            else:
                self.metric = datasets.load_metric(self.hparams.metric)
        else:
            self.metric = datasets.load_metric('glue', self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    def forward(self, **inputs):
        return self.model(**inputs)

    def predict(self, sentence):
        # Either encode single sentence or sentence pairs
        if len(self.data.text_fields) > 1:
            texts_or_text_pairs = []
            for text_field in self.data.text_fields:
                texts_or_text_pairs.append(sentence[text_field])
        else:
            texts_or_text_pairs = sentence[self.data.text_fields[0]]

        inputs = self.data.tokenizer.encode_plus(
                texts_or_text_pairs,
                return_tensors="pt")


        predictions = self.forward(**inputs)[0]

        if self.hparams.task_name == 'stsb':
            return {"result": predictions.numpy()[0][0]}
        else:
            return {"result": torch.argmax(predictions, axis=1).numpy()[0]}


    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.data.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.data.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == 'mnli':
            combined_score = []
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.eval_splits[i].split('_')[-1]
                preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x['loss'] for x in output]).mean()
                self.log(f'val_loss_{split}', loss, prog_bar=True)

                split_metrics = {f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()}
                combined_score.append(list(split_metrics.values()))
                self.log_dict(split_metrics, prog_bar=True)

            split_metrics["combined_score"] = np.mean(combined_score).item()
            self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)

        result = self.metric.compute(predictions=preds, references=labels)

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        if self.hparams.task_name == 'stsb':
            result["mse"] = ((preds - labels) ** 2).mean().item()

        self.log_dict(result, prog_bar=True)
        return loss

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Transformer")
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name_or_path", default="distilbert-base-cased", type=str, help="Encoder model to be used.")
        parser.add_argument("--train_data", default=None, type=str, help="Path to the file containing the train data. Example: data/cola/train.csv")
        parser.add_argument("--dev_data", default=None, type=str, help="Path to the file containing the dev data. Example: data/cola/dev.csv")
        parser.add_argument("--test_data", default=None, type=str, help="Path to the file containing the test data. Example: data/cola/test.csv")
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser
