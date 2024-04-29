# coding: utf-8
from torch import optim
import torch.nn as nn
import argparse

from utils import Parameters
from Preprocess import prepare_data
from Encoder import EncoderRNN
from Decoder import AttnDecoderRNN
from train import train_iters
from evaluate import evaluate, evaluate_randomly, evaluate_and_show_attention
from utils import load_model

# Command-line arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument('epochs', type=int)
parser.add_argument('--maxlength', type=int, default=70)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hiddensize', type=int, default=500)
parser.add_argument('--cuda', action="store_true")
parser.add_argument('--load', action="store_true")
args = parser.parse_args()

# Dictionary tokens
SOS_token = 0
EOS_token = 1
UNK_token = 2

# Training hyperparameters
hidden_size = args.hiddensize
n_layers = args.layers
dropout_p = 0.05
learning_rate = 0.0001
n_epochs = args.epochs

# Plot Parameters
plot_every = 1000
print_every = 1000


dataset = 'anki-tatoeba'

# Choice of computing device
if args.cuda:
    print('Using CUDA')
else:
    print('NOT Using CUDA. This can be slow.')

# Preprocessing of dataset
input_lang, output_lang, pairs = prepare_data(None, None, dataset, 'train', args.maxlength)
input_lang, output_lang, val_pairs = prepare_data(input_lang, output_lang, dataset, 'val', args.maxlength)
input_lang, output_lang, test_pairs = prepare_data(input_lang, output_lang, dataset, 'test', args.maxlength)

# Encapsulation of fundamental Parameters
pars = Parameters(input_lang, output_lang, args.maxlength, EOS_token,
                  SOS_token, UNK_token, args.cuda, print_every, plot_every,
                  n_epochs, learning_rate)

# Instantiation of Encoder and Decoder
encoder = EncoderRNN(input_lang.n_words, hidden_size, pars, n_layers)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, pars, n_layers, dropout_p=dropout_p)

# Loading of previously trained models
if args.load:
    load_model(encoder, 'encoder')
    load_model(decoder, 'decoder')

# Migrating data to GPU
if pars.USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Training of a model
if not args.load:
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, pars, pairs, val_pairs)

# Translate some random sentence
for _ in range(5):
    evaluate_randomly(encoder, decoder, pars, test_pairs)


# translate a sentence and save attention weights matrix
evaluate_and_show_attention("i love the city of the pictures .", encoder, decoder, pars)
