import argparse
from os import path

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default="train", type=str, help='Change the mode train | generate')
    parser.add_argument('--seed', default=9, type=int, help='Random seed for reproduce the result')
    parser.add_argument('--debug', default=True, type=bool, help='In debug, use small text file to generate corpus')

    parser.add_argument('--data_dir', default=path.join("data", "wikitext-2"), type=str,
                        help='The directory contains the data')
    parser.add_argument('--save_dir', default="log", type=str, help="Where to save the model")

    parser.add_argument('--batch_size', default=20, type=int, help='The batch size (first dimension of each data batch)')
    parser.add_argument('--init_lr', default=0.5, type=float, help='Initial learning rate for Adadelta')

    parser.add_argument('--embed_dim', default=100, type=int, help='The dimension of embedding')
    parser.add_argument('--drop', default=0.5, type=float, help='The dropout rate for other parts like embedding')
    parser.add_argument('--hid_dim', default=20, type=int, help='The dimension of hidden status of RNN')
    parser.add_argument('--layer_num', default=3, type=int, help='The number of layers of encoder RNN')
    parser.add_argument('--rnn', default='LSTM', type=str, help='The type of encoder RNN is LSTM | GRU')
    parser.add_argument('--rnn_drop', default=0.3, type=float, help='The dropout rate of encoder RNN')
    parser.add_argument('--rnn_bidir', default=True, type=bool, help='Whether to use bidirectional RNN')
    parser.add_argument('--bptt_len', default=35, type=int, help='The length to do Back Propagation Through Time')
    parser.add_argument('--tie_weights', default=False, type=bool,
                        help='Whether to tie the weights as described in https://arxiv.org/abs/1608.05859')

    parser.add_argument('--epoch_num', default=20, type=int, help='The number of epoch in training')
    parser.add_argument('--eval_span', default=4, type=int, help='Plot the loss and jot down every n steps')
    parser.add_argument('--print_span', default=2, type=int, help='Print info during training every n steps')
    parser.add_argument('--plot_span', default=1, type=int, help='Plot the loss and jot down every n steps')
    parser.add_argument('--checkpoint', default=5, type=int, help='Save the checkpoint of model every n steps')

    parser.add_argument('--generate_word_num', default=100, type=int, help='Generate how many words')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature when generate the text')

    opt = parser.parse_args()
    return opt
