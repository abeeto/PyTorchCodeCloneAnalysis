import os.path

import torch
import pickle as pkl
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import BertTokenizer

UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'


def get_vocab(args):
    if os.path.exists(args.vocab_path):
        vocab = pkl.load(open(args.vocab_path, 'rb'))
    else:
        raise
    args.vocab_nums = len(vocab)
    return vocab


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def my_collate_fn(x):

    tokens_id = torch.LongTensor([x_['tokens_id'] for x_ in x])
    labels = torch.LongTensor([x_['label'] for x_ in x])
    mask = torch.LongTensor([x_['mask'] for x_ in x])

    return tokens_id, labels, mask


def pad_size(tokens, args):
    seq_len = len(tokens)

    if args.pad_size:
        if len(tokens) < args.pad_size:
            tokens.extend([PAD] * (args.pad_size - seq_len))
            mask = [1] * seq_len + [0] * (args.pad_size - seq_len)
        else:
            tokens = tokens[:args.pad_size]
            mask = [1] * args.pad_size
            seq_len = args.pad_size

    return tokens, mask


def build_dataset(args, file_path, vocab):
    if args.model_name == 'Bert':
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    else:
        if args.use_word:
            tokenizer = lambda x: x.split(' ')
        else:
            tokenizer = lambda x: [y for y in x]

    instances = []

    with open(file_path, 'r') as fr_file:
        for line in tqdm(fr_file):
            info = line.strip()

            if not info:
                continue

            content, label = info.split('\t')

            if args.model_name == 'Bert':
                tokens = tokenizer.tokenize(content)
                tokens, mask = pad_size([CLS] + tokens, args)
                tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            else:
                tokens_id = []
                tokens = tokenizer(content)
                tokens, mask = pad_size(tokens, args)
                for word in tokens:
                    tokens_id.append(vocab.get(word, vocab.get(UNK)))

            # seq_len = len(tokens)
            #
            # if args.pad_size:
            #     if len(tokens) < args.pad_size:
            #         tokens.extend([PAD] * (args.pad_size - seq_len))
            #     else:
            #         tokens = tokens[:args.pad_size]
            #         seq_len = args.pad_size

            dict_instance = {
                'label': int(label),
                'tokens': tokens,
                'tokens_id': tokens_id,
                'mask': mask
            }

            instances.append(dict_instance)

    return instances


def all_macro(yhat_mac, y_mac):
    macro_acc = accuracy_score(y_mac, yhat_mac)
    macro_f1 = f1_score(y_mac, yhat_mac, average='macro')
    macro_p = precision_score(y_mac, yhat_mac, average='macro')
    macro_r = recall_score(y_mac, yhat_mac, average='macro')
    return macro_acc, macro_p,  macro_r, macro_f1


def all_micro(yhat_mic, y_mic):
    micro_acc = accuracy_score(y_mic, yhat_mic)
    micro_f1 = f1_score(y_mic, yhat_mic, average='micro')
    micro_p = precision_score(y_mic, yhat_mic, average='micro')
    micro_r = recall_score(y_mic, yhat_mic, average='micro')
    return micro_acc, micro_p, micro_r, micro_f1


def all_metrics(yhat, y):
    """
    :param yhat:
    :param y:
    :return:
    """
    names = ['acc', 'prec', 'rec', 'f1']
    macro_metrics = all_macro(yhat, y)

    y_mic = y.ravel()
    yhat_mic = yhat.ravel()
    micro_metrics = all_micro(yhat_mic, y_mic)

    metrics_dict = {names[i] + "_macro": macro_metrics[i] for i in range(len(macro_metrics))}
    metrics_dict.update({names[i] + '_micro': micro_metrics[i] for i in range(len(micro_metrics))})

    return metrics_dict


def early_stop(metrics_criterion, criterion, patience):
    if not np.all(np.isnan(metrics_criterion)):
        if len(metrics_criterion) >= patience:
            if criterion == 'loss_dev':
                return np.nanargmin(metrics_criterion) < len(metrics_criterion) - patience
            else:
                return np.nanargmax(metrics_criterion) < len(metrics_criterion) - patience
    else:
        return False


def print_metrics(metrics_test):
    print("[MACRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_macro"], metrics_test["prec_macro"], metrics_test["rec_macro"], metrics_test["f1_macro"]))

    print("[MICRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_micro"], metrics_test["prec_micro"], metrics_test["rec_micro"], metrics_test["f1_micro"]))
    # print()


def save_results(args, model, model_dir, best_scores):
    save_model = False
    if len(best_scores) == 1:
        save_model = True
    else:
        if args.criterion == 'dev_loss' and best_scores[-1] < min(best_scores[:-1]):
            save_model = True
        elif best_scores[-1] > max(best_scores[:-1]):
            save_model = True

    if save_model:
        sd = model.cpu().state_dict()
        torch.save(sd, model_dir + "/model_best_%s.pth" % args.criterion)
        if args.gpu >= 0:
            model.cuda(args.gpu)

        print("saved params, model to directory %s\n" % (model_dir))
