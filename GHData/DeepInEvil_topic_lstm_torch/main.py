from __future__ import print_function

import time
import gc
import os
import argparse

import numpy as np
from sklearn.externals import  joblib
import torch
from torch import nn
import torch.backends.cudnn as cudnn

from vocab import  VocabBuilder
from dataloader import TextClassDataLoader
from model import RNN, RNNTopic
from util import AverageMeter, accuracy
from util import adjust_learning_rate
from sklearn.metrics import accuracy_score
from gensim import models
import gensim

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=50*2, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=50, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=64, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=2, type=int, metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=3, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--early-stopping', default=3, type=int, help='early stopping on validation set')
parser.add_argument('--fc-layer', default=100, type=int, help='fully connected size after lstm')
parser.add_argument('--emb-drop', default=0.8, type=float, help='embeddidng dropout')
parser.add_argument('--mit-topic', default=False, help='with or without topic embedding')

args = parser.parse_args()
domains = ['kitchen', 'electronics', 'dvd', 'books']


def load_glove(path):
    """
    creates a dictionary mapping words to vectors from a file in glove format.
    """
    with open(path) as f:
        glove = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove[word] = vector
        return glove


def load_glove_embeddings(path, word2idx, embedding_dim=50):
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()


def get_lda_vec(lda_dict):
    """
    get lda vector
    :param lda_dict:
    :return:
    """
    lda_vec = np.zeros(50, dtype='float32')
    for id, val in lda_dict:
        lda_vec[id] = val
    return lda_vec


def get_id2word(idx, idx2w_dict):
    """
    get id2word mappings
    :param idx:
    :param idx2w_dict:
    :return:
    """
    try:
        return idx2w_dict[idx]
    except KeyError:
        return '__UNK__'


def get_theta(texts, lda, dictionari, idx2word):
    """
    get doc-topic distribution vector for all reviews
    :param texts:
    :param lda:e
    :param dictionari:
    :param idx2word:
    :return:
    """
    #print (texts[0])
    texts = [[get_id2word(idx, idx2word) for idx in sent] for sent in texts]
    #print (texts)
    review_alphas = np.array([get_lda_vec(lda[dictionari.doc2bow(sentence)]) for sentence in texts])
    #print (review_alphas)
    return torch.from_numpy(review_alphas)


def earlystop(val_acc_list, current_val_acc):
    """
    early stopping if accuracy doesn't increases after n steps
    :param val_acc_list:
    :param current_val_acc:
    :return:
    """
    #print (current_val_acc, best_val_acc)
    if len(val_acc_list) > args.early_stopping:
        best_val_acc = np.max(val_acc_list[-args.early_stopping:])
        print (current_val_acc, best_val_acc)
        if current_val_acc >= best_val_acc:
            return False
        else:
            return True
    else:
        return False


def train(train_loader, model, criterion, optimizer, epoch, lda_model, lda_dictionary, word2id):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    correct = 0.0
    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(train_loader):
        #print (input, target)
        data_time.update(time.time() - end)

        inp_topic = get_theta(input.numpy(), lda_model, lda_dictionary, word2id).cuda()
        inp_topic = inp_topic.type(torch.cuda.FloatTensor)
        #print (inp_topic[0])
        if args.cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        topic_var = torch.autograd.Variable(inp_topic, requires_grad = False)
        input_var = torch.autograd.Variable(input, requires_grad = False)
        target_var = torch.autograd.Variable(target, requires_grad = False)

        # compute output
        if args.mit_topic:
            output = model(input_var, topic_var)
        else:
            output = model(input_var)

        loss = criterion(output, target_var)
        out = (torch.max(output, 1))[1].cpu()
        #print (out)

        # measoure accuracy and record loss
        #correct += (out.data.numpy() == target).sum()
        #prec1 = 100 * correct / len(target_var)
        prec1 = accuracy_score(target, out.data.numpy())
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            gc.collect()


def validate(val_loader, model, criterion, lda_model, lda_dictionary, word2id):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0.0
    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(val_loader):
        inp_topic = get_theta(input.numpy(), lda_model, lda_dictionary, word2id).cuda()
        inp_topic = inp_topic.type(torch.cuda.FloatTensor)
        if args.cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        topic_var = torch.autograd.Variable(inp_topic, requires_grad = False)
        input_var = torch.autograd.Variable(input, requires_grad = False)
        target_var = torch.autograd.Variable(target, requires_grad = False)
        # compute output
        if args.mit_topic:
            output = model(input_var, topic_var)
        else:
            output = model(input_var)
        loss = criterion(output, target_var)
        out = (torch.max(output, 1))[1].cpu()
        # measure accuracy and record loss
        #correct += (out.data.numpy() == target).sum()
        #prec1 = 100 * correct / len(target_var)
        prec1 = accuracy_score(target, out.data.numpy())
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i!= 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            gc.collect()

    print(' * Accuracy {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def test(test_loader, model, criterion, lda_model, lda_dictionary, word2id):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0.0
    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(test_loader):
        inp_topic = get_theta(input.numpy(), lda_model, lda_dictionary, word2id).cuda()
        inp_topic = inp_topic.type(torch.cuda.FloatTensor)
        if args.cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        #inp_topic = torch.zeros(input.size(0), 50).uniform_(0, 1).cuda()
        topic_var = torch.autograd.Variable(inp_topic, requires_grad = False)
        input_var = torch.autograd.Variable(input, requires_grad = False)
        target_var = torch.autograd.Variable(target, requires_grad = False)
        # compute output
        if args.mit_topic:
            output = model(input_var, topic_var)
        else:
            output = model(input_var)
        loss = criterion(output, target_var)
        out = (torch.max(output, 1))[1].cpu()
        # measure accuracy and record loss
        #correct += (out.data.numpy() == target).sum()
        #prec1 = 100 * correct / len(target_var)
        prec1 = accuracy_score(target, out.data.numpy())
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i!= 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            gc.collect()

    print(' * Accuracy {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def run_model(domain):
    # create vocab
    print("===> creating vocabs for domain..." + domain)
    end = time.time()
    domain_d = 'reviews/leave_out_' + domain
    lda_model = models.LdaModel.load(domain_d + '/lda_model/lda_' + domain)
    lda_dict = gensim.corpora.Dictionary.load(domain_d + '/lda_model/dict_' + domain)
    print (domain_d)
    v_builder = VocabBuilder(path_file=domain_d + '/train.csv', min_sample=args.min_samples)
    d_word_index = v_builder.get_word_index()
    vocab_size = len(d_word_index)
    word2id = {v: k for k, v in d_word_index.iteritems()}
    #print (word2id)
    embeddings = load_glove_embeddings('/home/DebanjanChaudhuri/topic_lstm_torch/word_vecs/glove.6B.50d.txt', d_word_index)
    if not os.path.exists('gen_' + domain):
        os.mkdir('gen_' + domain)

    joblib.dump(d_word_index, 'gen_' + domain +'/d_word_index.pkl', compress=3)
    print('===> vocab creating: {t:.3f}'.format(t=time.time() - end))

    # create trainer
    print("===> creating dataloaders ...")
    end = time.time()
    train_loader = TextClassDataLoader(domain_d + '/train.csv', d_word_index, batch_size=args.batch_size)
    val_loader = TextClassDataLoader(domain_d +'/val.csv', d_word_index, batch_size=args.batch_size)
    test_loader = TextClassDataLoader(domain_d + '/test.csv', d_word_index, batch_size=args.batch_size)
    print('===> Dataloader creating: {t:.3f}'.format(t=time.time() - end))

    # create model
    print("===> creating rnn model ...")
    if args.mit_topic:
        print ("with topic vectors.")
        model = RNNTopic(vocab_size=vocab_size, embed_size=args.embedding_size,
                         num_output=args.classes, topic_size=50, hidden_size=args.hidden_size,
                         num_layers=args.layers, batch_first=True, use_gpu=args.cuda, embeddings=embeddings, emb_drop=args.emb_drop, fc_size=args.fc_layer)
    else:
        model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size,
                    num_output=args.classes, hidden_size=args.hidden_size,
                    num_layers=args.layers, batch_first=True, use_gpu=args.cuda, embeddings=embeddings, emb_drop=args.emb_drop, fc_size=args.fc_layer)

    print(model)

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    print(optimizer)
    print(criterion)

    if args.cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        model.cuda()
        criterion = criterion.cuda()

    #List for checking early stopping
    val_acc = []
    for epoch in range(1, args.epochs+1):

        adjust_learning_rate(args.lr, optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, lda_model, lda_dict, word2id)
        print ("getting performance on validation set!")
        v_acc = validate(val_loader, model, criterion, lda_model, lda_dict, word2id)
        print (len(val_acc), args.early_stopping)
        #if len(val_acc) > args.early_stopping:
        print ("checking early stopping.")
        if earlystop(val_acc, v_acc):
            print ("Early stopping!")
            break
        val_acc.append(v_acc)

        # save current model
        if epoch % args.save_freq == 0:
            name_model = 'rnn_{}.pkl'.format(epoch)
            path_save_model = os.path.join('gen_'+ domain + '/', name_model)
            joblib.dump(model.float(), path_save_model, compress=2)
    print ("Results on test set for leave-out-domain!" + domain)
    test_acc = test(test_loader, model, criterion, lda_model, lda_dict, word2id)
    return test_acc


if __name__ == '__main__':
    domain_acc_dict = {}
    # training and testing
    for domain in domains:
        domain_acc_dict[domain] = run_model(domain)

    print ("Reporting final accuracies on testing set:")
    print (domain_acc_dict)
