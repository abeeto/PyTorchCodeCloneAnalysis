import torch
from torchtext import data
import os
import random
torch.set_num_threads(1)
torch.manual_seed(233)
random.seed(233)
torch.cuda.manual_seed(233)
torch.backends.cudnn.enabled = False

import argparse
import datetime
import classification_datasets
import data_five

import Bi_LSTM
import train_lstm
import train_cnn

parser = argparse.ArgumentParser(description='LSTM text classificer')
# argument
# parser.add_argument()
parser.add_argument('-epochs', type=int, default=256)
parser.add_argument('-batch-size', type=int, default=64)
parser.add_argument('-log-interval', type=int, default=1)
parser.add_argument('-test-interval', type=int, default=100)
parser.add_argument('-save-interval', type=int, default=100)
parser.add_argument('-save-dir', type=str, default='snapshot')
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-dropout-embed', type=float, default=0.5)
parser.add_argument('-dropout-model', type=float, default=0.5)
parser.add_argument('-device', type=int, default=-1)
parser.add_argument('-hidden-dim', type=int, default=300)
parser.add_argument('-embedding-dim', type=int, default=300)
parser.add_argument('-use-pretrained-emb', action='store_true', default=True)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-class-five', action='store_true', default=False)
# model
parser.add_argument('-lstm', action='store_true', default=False)
parser.add_argument('-bilstm', action='store_true', default=True)

parser.add_argument('-train-cnn', action='store_true', default=False)
parser.add_argument('-use-cuda', action='store_true', default=False)
args = parser.parse_args()

# load dataset
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

if args.class_five:
    train_iter, dev_iter, test_iter = data_five.load_mr(text_field, label_field, batch_size=args.batch_size)
else:
    train_iter, dev_iter, test_iter = classification_datasets.load_mr(text_field, label_field, batch_size=args.batch_size)

args.word_dict = text_field.vocab.freqs
# print(args.word_dict)
args.word_list = text_field.vocab.itos
# print(args.word_list)
# padID = args.word_dict['<pad>']
# print(padID)

# wv_cat = loader.vector_loader(count_words_reset)
# pretrained_weight = wv_cat
# args.pretrained_weight = pretrained_weight

# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# model
if args.bilstm:
    model = Bi_LSTM.BiLSTM(args)
    if args.use_cuda:
        model = model.cuda()

if os.path.exists("./Test_Result.txt"):
    os.remove("./Test_Result.txt")

# train
print("Training start")
if args.train_cnn:
    train_cnn.train(train_iter, dev_iter, test_iter, model, text_field, label_field, args)
else:
    train_lstm.train(train_iter, dev_iter, test_iter, model, text_field, label_field, args)

