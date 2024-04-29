import argparse
import json
import os
import sys

import torch
from torch import nn

from model import RNNModel
from train import train_novel
from train import predict_novel
from data_preprocess import load_data_novel

local_rank = int(os.environ["LOCAL_RANK"])

def get_rnn_layer(net, vocab_size, num_hiddens, num_layers):
    if net == 'GRU':
        return nn.GRU(vocab_size, num_hiddens, num_layers)
    elif net == 'LSTM':
        return nn.LSTM(vocab_size, num_hiddens, num_layers, dropout=0.5)
    else:
        print('unrecognized net, use RNN as default')
        return nn.RNN(vocab_size, num_hiddens, num_layers)

def save_model(args, model):
    save_model_name = args.save_model_name
    if save_model_name:
        file_list = os.listdir('model/')
        new_dir = 'model/'+str(len(file_list))
        os.mkdir(new_dir)
        with open(new_dir + '/train_args.json', 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f, ensure_ascii=False)
        try:
            torch.save(model.state_dict(), new_dir + '/' + save_model_name)
        except Exception as e:
            print('save model error', e)
            print('the state dict of model is\n', model.state_dict())
        else:
            print('finish saving model.')

def load_model(load_model_name, model):
    if load_model_name:
        try:
            model_state_dict = torch.load(load_model_name)
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print('load model error', e)
            sys.exit()
    return model

def to_train(args):
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    batch_size, time_steps, max_tokens = args.batch_size, args.num_steps, args.max_token
    token, language = args.token, args.language

    train_iter, vocab = load_data_novel(batch_size, time_steps, token, language, max_tokens)

    vocab_size, num_hiddens, num_layers = len(vocab), args.num_hiddens, args.num_layers
    lr, num_epochs, net = args.lr, args.num_epochs, args.net
    rnn_layer = get_rnn_layer(net, vocab_size, num_hiddens, num_layers)
    model = RNNModel(rnn_layer, vocab_size).to(device)
    load_model_name = args.load_model_name
    model = load_model(load_model_name, model)
    # print('start training model. The training args are:\n', args.__dict__)
    train_novel(model, local_rank, train_iter, lr, num_epochs, device)
    
    save_model(args, model)

def to_predict(args):
    prefix = args.prefix
    num_preds = args.num_preds
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model_name = args.load_model_name
    model_dir = load_model_name.rsplit('/', 1)[0]
    with open(model_dir + '/train_args.json', 'r', encoding='utf-8') as f:
        train_args = json.load(f)
    _, vocab = load_data_novel(train_args['batch_size'], train_args['num_steps'], train_args['token'], train_args['language'], train_args['max_token'])
    rnn_layer = get_rnn_layer(train_args['net'], len(vocab), train_args['num_hiddens'], train_args['num_layers'])
    model = RNNModel(rnn_layer, len(vocab)).to(device)
    model = load_model(load_model_name, model)
    res = predict_novel(prefix, num_preds, model, vocab, device)
    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='rank of distributed processes')
    subparser = parser.add_subparsers()
    train_parser = subparser.add_parser('train', help='training the model')
    train_parser.add_argument('--language', type=str, default='chinese')
    train_parser.add_argument('--token', type=str, default='char')
    train_parser.add_argument('--net', type=str, default='GRU', help='which rnn to use GRU/LSTM')
    train_parser.add_argument('--batch_size', type=int, default=256)
    train_parser.add_argument('--num_steps', type=int, default=35)
    train_parser.add_argument('--max_token', type=int, default=1000000, help='how many tokens for training')    
    train_parser.add_argument('--num_hiddens', type=int, default=256)    
    train_parser.add_argument('--num_layers', type=int, default=1)    
    train_parser.add_argument('--num_epochs', type=int, default=500)
    train_parser.add_argument('--lr', type=float, default=1e-3) 
    train_parser.add_argument('--load_model_name', type=str, default=None)
    train_parser.add_argument('--save_model_name', type=str, default=None)
    train_parser.set_defaults(action='train')

    predict_parser = subparser.add_parser('predict', help='predict the from the prefix')
    predict_parser.add_argument('--load_model_name', type=str, default=None, required=True)
    predict_parser.add_argument('--prefix', type=str, default="叶凡")
    predict_parser.add_argument('--num_preds', type=int, default=1000)
    predict_parser.set_defaults(action='predict')

    args = parser.parse_args()
    
    if args.action == 'train':
        to_train(args)
    elif args.action == 'predict':
        to_predict(args)