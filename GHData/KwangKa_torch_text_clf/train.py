# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 15:22
# @Author  : kaka

import argparse
import os
from pathlib import Path
import json
import pickle
import torch
import torch.nn.functional as F
from model.textcnn import TextCNN
from model.textrnn import TextRNN
from model.transformer import TransformerClf
import dataset


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_name", default="textcnn", type=str, help="Model name.")
    parser.add_argument("--output_model_path", default="./output_models/", type=str, help="Path of the output model.")
    parser.add_argument("--data_path", default="./data/", type=str, help="Path of the dataset.")
    parser.add_argument("--config_path", default="./conf/", type=str, help="Path of the config file.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_seq_length", type=int, default=100, help="Max sequence length.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--early_stopping", type=int, default=100, help="Early stop.")
    parser.add_argument("--display_interval", type=int, default=1, help="Display interval.")
    parser.add_argument("--val_interval", type=int, default=30, help="Validation interval.")

    args = parser.parse_args()
    return args


def train_model(train_iter, val_iter, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_idx = 0
    best_acc = 0
    best_step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            model.train()
            feature, target = batch[0], batch[1]
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            step_idx += 1
            if step_idx % args.display_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                b_size = len(target)
                train_acc = 100.0 * corrects / b_size
                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                    step_idx,
                    loss.item(),
                    train_acc,
                    corrects,
                    b_size),
                    end='\r',
                    flush=True
                )
            if step_idx % args.val_interval == 0:
                val_acc = eval_model(val_iter, model)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_step = step_idx
                    print('Saving best model, acc: {:.4f}%'.format(best_acc))
                    save_model(model, args.output_model_path)
                else:
                    if step_idx - best_step >= args.early_stopping:
                        print('early stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        return


def eval_model(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch[0], batch[1]
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
        avg_loss,
        accuracy,
        corrects,
        size)
    )
    return accuracy


def save_vocab(vocab, fname):
    with open(fname, 'wb') as h:
        pickle.dump(vocab, h)


def save_model(model, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = '{0}/model.pt'.format(save_dir)
    torch.save(model.state_dict(), save_path)


def load_model_conf(args):
    f_conf = Path(args.config_path, "{0}_conf.json".format(args.model_name))
    with open(f_conf, 'r', encoding='utf8') as h_in:
        model_conf = json.load(h_in)
    return model_conf


def build_model(args, model_conf, vocab_size):
    if args.model_name == "textcnn":
        model = TextCNN(
            vocab_size=vocab_size,
            class_num=model_conf['class_num'],
            embed_size=model_conf['embed_size'],
            kernel_num=model_conf['kernel_num'],
            kernel_sizes=model_conf['kernel_sizes'],
            dropout_rate=model_conf['dropout_rate']
        )
    elif args.model_name == "textrnn":
        model = TextRNN(
            vocab_size=vocab_size,
            class_num=model_conf['class_num'],
            embed_size=model_conf['embed_size'],
            hidden_size=model_conf['hidden_size'],
            num_layers=model_conf['num_layers'],
            bidirectional=model_conf['bidirectional'],
            dropout_rate=model_conf['dropout_rate']
        )
    elif args.model_name == "transformer":
        model = TransformerClf(
            vocab_size=vocab_size,
            class_num=model_conf['class_num'],
            max_seq_length=model_conf['max_seq_length'],
            embed_size=model_conf['embed_size'],
            nhead=model_conf['nhead'],
            num_layers=model_conf['num_layers'],
            dim_feedforward=model_conf['dim_feedforward'],
            dropout_rate=model_conf['dropout_rate']
        )
    else:
        raise Exception("Invalid model name:{0}".format(args.model_name))
    return model


def main():
    args = parse_args()
    if args.model_name not in ["textcnn", "textrnn", "transformer"]:
        raise Exception("Invalid model name:{0}".format(args.model_name))

    train_iter, val_iter, test_iter, vocab = dataset.get_data(args)
    save_vocab(vocab, Path(args.output_model_path, './vocab.pkl'))
    vocab_size = len(vocab)

    model_conf = load_model_conf(args)
    if "max_seq_length" in model_conf and model_conf["max_seq_length"] != args.max_seq_length:
        raise Exception("Data padding max length != model max seq length")

    model = build_model(args, model_conf, vocab_size)
    print("===== model structure:")
    print(model)
    print("=====")

    train_model(train_iter=train_iter, val_iter=val_iter, model=model, args=args)

    print('\n\nTest with best model')
    model.load_state_dict(torch.load(Path(args.output_model_path, './model.pt')))
    eval_model(data_iter=test_iter, model=model)


if __name__ == '__main__':
    main()
