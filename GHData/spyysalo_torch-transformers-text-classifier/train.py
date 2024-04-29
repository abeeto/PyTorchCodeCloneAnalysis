#!/usr/bin/env python

import os
import sys
import logging

from argparse import ArgumentParser

from torch import nn
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from modeling import T5ForSequenceClassification


LABEL_PREFIX = '__label__'

LABEL_STRINGS = 'label_str'

DEFAULTS = {
    'DATA_DIR': 'data',
    'TOKENIZER': 'tokenizer',
    'MODEL': 'model',
    'MAX_LENGTH': 256,
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 16,
    'EPOCHS': 2,
}

def argparser():
    ap = ArgumentParser()
    ap.add_argument(
        '--tokenizer',
        metavar='DIR',
        default=DEFAULTS['TOKENIZER']
    )
    ap.add_argument(
        '--model',
        metavar='DIR',
        default=DEFAULTS['MODEL']
    )
    ap.add_argument(
        '--data',
        metavar='DIR',
        default=DEFAULTS['DATA_DIR']
    )
    ap.add_argument(
        '--max_length',
        type=int,
        default=DEFAULTS['MAX_LENGTH']
    )
    ap.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULTS['BATCH_SIZE']
    )
    ap.add_argument(
        '--epochs',
        type=int,
        default=DEFAULTS['EPOCHS']
    )
    ap.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULTS['LEARNING_RATE']
    )
    return ap


class MultilabelTrainer(Trainer):
    # Following https://huggingface.co/transformers/main_classes/trainer.html
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def load_data(fn):
    texts, label_strings = [], []
    with open(fn) as f:
        for ln, line in enumerate(f, start=1):
            labels = []
            text = line.strip()
            while text.startswith(LABEL_PREFIX):
                label, text = text.split(None, 1)
                label = label.replace('__label__', '')
                labels.append(label)
            texts.append(text)
            label_strings.append(labels)
    dataset = Dataset.from_dict({
        LABEL_STRINGS: label_strings,
        'text': texts,
    })
    return dataset


def load_datasets(directory):
    datasets = DatasetDict()
    for s in ('train', 'dev', 'test'):
        datasets[s] = load_data(os.path.join(directory, f'{s}.txt'))
    return datasets


def get_labels(data):
    labels = set()
    for d in data.values():
        for l in d[LABEL_STRINGS]:
            labels.update(l)
    return sorted(labels)


def is_multiclass(data):
    for d in data.values():
        for l in d[LABEL_STRINGS]:
            if len(l) != 1:
                return False
    return True


def load_model(directory, num_labels, args):
    if 't5' in directory.lower():
        # special case for T5, which isn't supported by
        # AutoModelForSequenceClassification
        logging.warning(
            f'assuming {directory} is T5 model. T5 support is experimental '
            f'and results may not reflect optimal model performance.'
        )
        class_ = T5ForSequenceClassification
    else:
        class_ = AutoModelForSequenceClassification

    model = class_.from_pretrained(
        directory,
        num_labels=num_labels
    )

    model.config.max_length = args.max_length
    return model


def load_tokenizer(directory, args):
    tokenizer = AutoTokenizer.from_pretrained(directory)
    if tokenizer.pad_token is None:
        pad = tokenizer.eos_token
        logging.warning(f'setting pad_token to {pad}')
        tokenizer.add_special_tokens({ "pad_token": pad })
    tokenizer.add_prefix_space = True
    tokenizer.model_max_length = args.max_length
    return tokenizer


def make_encode_text_function(tokenizer):
    def encode_text(example):
        encoded = tokenizer(
            example['text'],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        return encoded
    return encode_text


def make_encode_label_function(labels, multiclass):
    if multiclass:
        label_map = { l: i for i, l in enumerate(labels) }
        def encode_label(example):
            assert len(example[LABEL_STRINGS]) == 1
            example['label'] = label_map[example[LABEL_STRINGS][0]]
            return example
    else:
        mlb = MultiLabelBinarizer()
        mlb.fit([labels])
        def encode_label(example):
            example['label'] = mlb.transform([example[LABEL_STRINGS]])[0]
            return example
    return encode_label


def accuracy(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=1)
    return { 'accuracy': sum(y_pred == y_true) / len(y_true) }


def microf1(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions > 0
    return { 'microf1': f1_score(y_true, y_pred, average='micro') }


def main(argv):
    args = argparser().parse_args(argv[1:])

    data = load_datasets(args.data)
    labels = get_labels(data)
    multiclass = is_multiclass(data)
    print(f'multiclass: {multiclass}')

    tokenizer = load_tokenizer(args.tokenizer, args)

    encode_text = make_encode_text_function(tokenizer)
    encode_label = make_encode_label_function(labels, multiclass)

    data = data.map(encode_text)
    data = data.map(encode_label)

    model = load_model(args.model, len(labels), args)

    # This needs to be set explicitly for some reason
    model.config.pad_token_id = tokenizer.pad_token_id

    train_args = TrainingArguments(
        'output_dir',
        save_strategy='no',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )

    if multiclass:
        TrainerClass = Trainer
        metrics = accuracy
    else:
        TrainerClass = MultilabelTrainer
        metrics = microf1

    trainer = TrainerClass(
        model,
        train_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tokenizer,
        compute_metrics=metrics
    )

    trainer.train()

    print(trainer.evaluate(data['test'], metric_key_prefix='test'))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
