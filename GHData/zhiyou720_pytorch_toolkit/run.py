# coding: UTF-8
import time
import torch
import torch.backends.cudnn as bk
import argparse
import numpy as np
from importlib import import_module
from train_eval import train, init_network, test

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN,'
                                                             ' TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'

    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    if model_name == 'FastText':
        from utils_fasttext import build_data_set, build_iterator, get_time_dif

        embedding = 'random'
        vocab, train_data, dev_data, test_data = build_data_set(config, args.word)
    else:
        from utils import TrainDataGenerator, build_iterator, get_time_dif

        data_gen = TrainDataGenerator(config, args.word)
        vocab, train_data, dev_data, test_data = data_gen.vocab, data_gen.train_data, data_gen.valid_data, data_gen.test_data

    np.random.seed(1)
    torch.manual_seed(1)
    bk.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
    # test(config, model, test_iter)
