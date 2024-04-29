import argparse
import math
from model.concept_tagger import ConceptTagger
from tools.utils import getNERdata, readTokenEmbeddings, data_generator,  ExtractLabelsFromTokens
from tools.Log import Logger
import tools.conlleval as conlleval
import torch
import torch.nn as nn
import sys
import time
import numpy as np
import os


def model_config():
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument('--dataset', type=str, default='SNIPS')
    data_arg.add_argument('--data_dir', type=str, default='/home/sh/data/JointSLU-DataSet/formal_snips')
    data_arg.add_argument('--description_path', type=str, default='data/snips_slot_description.txt')
    data_arg.add_argument("--save_dir", type=str, default='/home/sh/code/CT_torch/data/')
    data_arg.add_argument("--embed_file", type=str, default='/home/sh/data/komninos_english_embeddings.gz')

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size1", type=int, default=256)
    net_arg.add_argument("--hidden_size2", type=int, default=128)
    net_arg.add_argument("--bidirectional", type=bool, default=True)
    net_arg.add_argument("--dropout", type=float, default=0.5)
    net_arg.add_argument("--crf", type=bool, default=True)

    # Training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--cross_domain", type=bool, default=True)
    train_arg.add_argument("--target_domain", type=str, default='BookRestaurant')
    train_arg.add_argument("--epoch", type=int, default=10)
    train_arg.add_argument("--log_every", type=int, default=50)
    train_arg.add_argument("--log_valid", type=int, default=200)
    train_arg.add_argument("--patience", type=int, default=5)
    train_arg.add_argument("--max_num_trial", type=int, default=5)
    train_arg.add_argument("--lr_decay", type=float, default=0.5)
    train_arg.add_argument("--learning_rate", type=float, default=0.001)
    train_arg.add_argument("--run_type", type=str, default='train')

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--device", type=str, default='cuda:1')
    misc_arg.add_argument("--batch_size", type=int, default=32)

    config = parser.parse_args()

    return config


def evaluate(model, data, batch_size, log):
    was_training = model.training
    model.eval()
    pred = []
    gold = []
    with torch.no_grad():
        for pa in data_generator(data, batch_size):
            x = pa[0]
            y = pa[1]
            slot = pa[2]
            _x, _y, p = model(x, y,slot, 'test')
            gold += _y
            pred += p


    if was_training:
        model.train()

    _gold = []
    _pred = []
    for i in gold:
        for j in i:
            _gold.append(j)
    for i in pred:
        for j in i:
            _pred.append(j)
    return conlleval.evaluate(_gold, _pred, log, verbose=True)


def train(config):
    dataDict = getNERdata(dataSetName=config.dataset,
                          dataDir=config.data_dir,
                          desc_path=config.description_path,
                          cross_domain=config.cross_domain,
                          target_domain=config.target_domain)

    emb, word2Idx = readTokenEmbeddings(config.embed_file)
    label2Idx = {'I':0,'O':1,'B':2}


    max_batch_size = math.ceil(len(dataDict['source']['train']) / config.batch_size)


    model = ConceptTagger(config, emb, word2Idx, label2Idx, dataDict['description'])
    model.train()
    model = model.to(config.device)
    hist_valid_scores = []
    patience = num_trial = 0
    train_iter = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_time = time.time()

    config.save_dir = config.save_dir + config.target_domain+'/'

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    if os.path.exists(os.path.join(config.save_dir, 'params')):
        os.remove(os.path.join(config.save_dir, 'params'))

    log = Logger(os.path.join(config.save_dir, '_src.txt'),level='info')

    for epoch in range(config.epoch):
        for da in data_generator(dataDict['source']['train'], config.batch_size):
            train_iter += 1
            x = da[0]
            y = da[1]
            slot = da[2]

            loss = model(x, y, slot, 'train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_iter % config.log_every == 0:
                # print(
                #     'epoch %d, iter %d, loss %.2f, time elapsed %.2f sec' %
                #     (epoch, train_iter, loss, time.time() - train_time),
                #     file=sys.stderr)
                log.logger.info(
                    'epoch %d, iter %d, loss %.2f, time elapsed %.2f sec' %
                    (epoch, train_iter, loss, time.time() - train_time))

            train_time = time.time()

            if train_iter % config.log_valid == 0:
                valid_metric_pre, valid_metric_rec, valid_metric_f1 = evaluate(model, dataDict['source']['dev'], config.batch_size, log)
                test_metric_pre, test_metric_rec, test_metric_f1 = evaluate(model, dataDict['source']['test'], config.batch_size, log)
                # print("val_pre : %.4f, val_rec : %.4f, val_f1 : %.4f" % (valid_metric_pre, valid_metric_rec, valid_metric_f1), file=sys.stderr)
                # print("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (test_metric_pre, test_metric_rec, test_metric_f1), file=sys.stderr)
                log.logger.info("val_pre : %.4f, val_rec : %.4f, val_f1 : %.4f" % (
                valid_metric_pre, valid_metric_rec, valid_metric_f1))
                log.logger.info("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (
                test_metric_pre, test_metric_rec, test_metric_f1))
                is_better = len(hist_valid_scores) == 0 or valid_metric_f1 > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric_f1)
                if is_better:
                    patience = 0
                    # print('save currently the best model to [%s]' % (config.save_dir + 'model'), file=sys.stderr)
                    log.logger.info('save currently the best model to [%s]' % (config.save_dir + 'model'))
                    model.save(config.save_dir + 'model')

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), config.save_dir + 'optim')
                elif patience < config.patience:
                    patience += 1
                    log.logger.info('hit patience %d' % patience)
                    # print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(config.patience):
                        num_trial += 1
                        log.logger.info('hit #%d trial' % num_trial)
                        # print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == config.max_num_trial:
                            log.logger.info('early stop!')
                            # print('early stop!', file=sys.stderr)
                            exit(0)


                        lr = optimizer.param_groups[0]['lr'] * config.lr_decay
                        log.logger.info('load previously best model and decay learning rate to %f' % lr)
                        # print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(config.save_dir +'model', map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(config.device)

                        log.logger.info('restore parameters of the optimizers')
                        # print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(config.save_dir + 'optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0


def cross_domain(model_path, device='cpu'):
    model = ConceptTagger.load(model_path, device)
    config = model.config
    log = Logger(os.path.join(config.save_dir, '_tgt.txt'), level='info')
    dataDict = getNERdata(dataSetName=config.dataset,
                          dataDir=config.data_dir,
                          desc_path=config.description_path,
                          cross_domain=config.cross_domain,
                          target_domain=config.target_domain)



    model.to(device)
    test_metric_pre, test_metric_rec, test_metric_f1 = evaluate(model, dataDict['target']['test'], config.batch_size, log)

    log.logger.info("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (test_metric_pre, test_metric_rec, test_metric_f1))



if __name__ == '__main__':
    config = model_config()
    run_type = config.run_type
    if run_type == "train":
        train(config)
    elif run_type == "test":
        cross_domain(config.save_dir + config.target_domain+'/model', config.device)


