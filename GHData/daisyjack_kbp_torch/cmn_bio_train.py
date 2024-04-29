from decoder import BahdanauAttnDecoderRNN
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm
from masked_cross_entropy import masked_cross_entropy
from configurations import to_np, Logger, get_conf
from encoder import LoadEmbedding, CNNEncoder
from bio_model import BioRnnDecoder, CMNBioCNNEncoder
from batch_getter import BatchGetter, get_source_mask, CMNBioBatchGetter
import codecs
import time
import random
import math
from bio_eval_ner import cmn_eval_all, batch_cmn_eval_all
from numpy import linalg as LA
import math
import os
import sys
import argparse
# import pydevd
#
# pydevd.settrace('10.214.129.230', port=31235, stdoutToServer=True, stderrToServer=True)
torch.manual_seed(1)

def schedule_samp_rate(iteration):
    k = 50
    if iteration < k * 200:
        rate = k / (k + math.exp(iteration / k))
    else:
        rate = 0
    return rate

def random_pick(some_list, probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item, item_probability in zip(some_list,probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def train_iteration(logger, config, my_arg, step, encoder, decoder, encoder_optimizer, decoder_optimizer, this_batch):
    # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['hidden_size']))
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    this_batch_num = len(this_batch[2])
    this_batch_max_target = max(this_batch[2])
    last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))
    print 'seq_length', max(this_batch[3]), 'label_length', this_batch_max_target  # (output_size, B, 1)

    data = Variable(this_batch[0])
    target = Variable(this_batch[1])
    target_length = Variable(torch.LongTensor(this_batch[2]))
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size']/2))  # encoder gru initial hidden state

    if config['USE_CUDA']:
        last_hidden = last_hidden.cuda(config['cuda_num'])
        word_input = word_input.cuda(config['cuda_num'])
        data = data.cuda(config['cuda_num'])
        target = target.cuda(config['cuda_num'])
        target_length = target_length.cuda(config['cuda_num'])
        h_0 = h_0.cuda(config['cuda_num'])

    encoder_outputs = encoder(step, data, h_0, this_batch[3])
    # encoder_outputs = encoder_outputs.transpose(1,2)
    # encoder_outputs = encoder_outputs.transpose(0,1)
    source_mask = Variable(get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['cuda_num'])
    encoder_outputs = encoder_outputs * source_mask





    # decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'], config['decoder_output_size'], config['decoder_layers'])
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # word_input = Variable(torch.LongTensor([[0], [1]]))
    # target = Variable(torch.LongTensor([[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]]))  # (batch, max_label_length)

    # length = Variable(torch.LongTensor([5,7]))

        # decoder.cuda(config['cuda_num'])


    # train
    decoder_out_label = []
    seq_label_prob = Variable(torch.zeros(this_batch_max_target, this_batch_num, config['decoder_output_size']))
    if config['USE_CUDA']:
        seq_label_prob = seq_label_prob.cuda(config['cuda_num'])

    rate = schedule_samp_rate(step)
    # rate=0
    for time_step in range(this_batch_max_target):
        label_logits, cur_hidden = decoder(step, word_input, last_hidden, encoder_outputs[time_step])
        last_hidden = cur_hidden
        seq_label_prob[time_step] = label_logits
        # Choose top word from label_prob
        # value, label = label_prob.topk(1)
        # decoder_out_label.append(label)
        # not teacher-forcing
        # word_input = label

        # teacher-forcing
        if my_arg == 0:
            word_input = target[:, time_step]
        else:
            # value, label = label_logits.data.topk(1)
            # decoder_out_label.append(label)
            # word_input = Variable(label)  # Chosen word is next input
            # if config['USE_CUDA']:
            #     word_input = word_input.cuda(config['cuda_num'])
            a = random_pick([0,1], [rate, 1-rate])
            if a == 0:
                word_input = target[:, time_step]
            else:
                value, label = label_logits.data.topk(1)
                # decoder_out_label.append(label)
                word_input = Variable(label)  # Chosen word is next input
                if config['USE_CUDA']:
                    word_input = word_input.cuda(config['cuda_num'])


    # decoder_prob = Variable(torch.FloatTensor([[[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]], [[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]]]))
    #
    # if config['USE_CUDA']:
    #     decoder_prob = decoder_prob.cuda(config['cuda_num'])

    loss = masked_cross_entropy(seq_label_prob.transpose(0,1).contiguous(), target, target_length)
    # loss = masked_cross_entropy(F.softmax(decoder_prob.transpose(0,1).contiguous()), target, length)
    print 'loss: ', loss.data[0]
    logger.scalar_summary('loss', loss.data[0], step)
    loss.backward()
    # e_before_step = [(tag, to_np(value)) for tag, value in encoder.named_parameters()]
    # d_before_step = [(tag, to_np(value)) for tag, value in decoder.named_parameters()]

    clip_grad_norm(decoder.parameters(), config['clip_norm'])
    clip_grad_norm(encoder.parameters(), config['clip_norm'])
    # for tag, value in encoder.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    # for tag, value in decoder.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    decoder_optimizer.step()
    encoder_optimizer.step()
    # e_after_step = [(tag, to_np(value)) for tag, value in encoder.named_parameters()]
    # d_after_step = [(tag, to_np(value)) for tag, value in decoder.named_parameters()]
    # for before, after in zip(e_before_step, e_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)
    #
    # for before, after in zip(d_before_step, d_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)


# another_decoder = BahdanauAttnDecoderRNN(encoder_outputs_dim, hidden_size, output_size, n_layers)
# another_decoder.load_state_dict(torch.load('net_params.pkl'))
# decoder_out_label = []
# seq_label_prob = Variable(torch.zeros(max_label_length, batch_size, output_size))
# word_input = Variable(torch.LongTensor([[0], [1]]))
# if USE_CUDA:
#     seq_label_prob = seq_label_prob.cuda()
#     word_input = word_input.cuda()
#     another_decoder.cuda()
# for time_step in range(max_label_length):
#     label_prob, cur_hidden, attn_weights = another_decoder(word_input, last_hidden, encoder_outputs)
#     last_hidden = cur_hidden
#     seq_label_prob[time_step] = label_prob
#     # Choose top word from label_prob
#     value, label = label_prob.topk(1)
#     decoder_out_label.append(label.data)
#     word_input = label
# print decoder_out_label

def train_epoch(logger, config, my_arg, epoch, ex_iterations, batch_getter, encoder, decoder, encoder_optimizer, decoder_optimizer):
    batch_getter.reset()
    for iteration, this_batch in enumerate(batch_getter):
        time0 = time.time()
        print 'epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration)
        train_iteration(logger, config, my_arg, ex_iterations + iteration, encoder, decoder, encoder_optimizer, decoder_optimizer, this_batch)
        time1 = time.time()
        print 'this iteration time: ', time1 - time0, '\n'
        if (ex_iterations + iteration) % config['save_freq'] == 0:
            torch.save(decoder.state_dict(), os.path.join(config['model_dir'], 'decoder_params.pkl'))
            torch.save(encoder.state_dict(), os.path.join(config['model_dir'], 'encoder_params.pkl'))
    return ex_iterations + iteration


def train(my_arg, log_dir, config, encoder, decoder, encoder_optimizer, decoder_optimizer):
    logger = Logger(log_dir)
    log_file = open(os.path.join(log_dir, 'eval_log'), 'w')
    batch_getter = CMNBioBatchGetter(config, config['train_data'], config['batch_size'], shuffle=True, bio=True)
    # batch_getter = BatchGetter('data/train.txt', 8)
    ex_iterations = 0
    f_max = 0
    low_epoch = 0
    for i in range(100000):
        # f, p, r = cmn_eval_all(config, log_dir, False)
        result = train_epoch(logger, config, my_arg, i, ex_iterations, batch_getter, encoder, decoder, encoder_optimizer, decoder_optimizer)
        ex_iterations = result + 1
        if config['epoch'] is None:
            f, p, r = batch_cmn_eval_all(config, log_dir, False)
            log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(i, f, p, r))
            log_file.flush()
            if f >= f_max:
                f_max = f
                low_epoch = 0
                os.system('cp {} {}'.format(os.path.join(config['model_dir'], 'decoder_params.pkl'),
                                            os.path.join(config['model_dir'], 'early_decoder_params.pkl')))
                os.system('cp {} {}'.format(os.path.join(config['model_dir'], 'encoder_params.pkl'),
                                            os.path.join(config['model_dir'], 'early_encoder_params.pkl')))
            else:
                low_epoch += 1
                log_file.write('low' + str(low_epoch) + '\n')
                log_file.flush()
            if low_epoch >= config['early_stop']:
                break
        else:
            if i >= config['epoch'] - 1:
                f, p, r = batch_cmn_eval_all(config, log_dir, False)
                log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(i, f, p, r))
                log_file.flush()
                break

    # f, p, r = cmn_eval_all(config, log_dir, False)
    # log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(i, f, p, r))
    # log_file.flush()
    log_file.close()



if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='eng, cmn, spa')  # cmn
    parser.add_argument('--my_arg', help='mode', type=int)  # 0
    parser.add_argument('--log_dir', help='log file dir')  # bio_cmn_logs0  OR  bio_cmn_logs1
    # data/cmn_train_extra+training.txt OR data/cmn_train_extra+training+eval.txt
    # data/cmn_train+dev_extra+training.txt OR data/cmn_train+dev_extra+training+eval.txt
    parser.add_argument('--train_data', help='train data')
    # data/cmn_dev_extra+training.txt   OR data/cmn_dev_extra+training+eval.txt
    parser.add_argument('--dev_data', help='dev data')
    parser.add_argument('--model_dir', help='model dir')  # cmn_bio_model0  OR  cmn_bio_model1
    parser.add_argument('--cuda_num', type=int, help='cuda num')
    # eng 22 OR 22
    parser.add_argument('--epoch', type=int, help='epochs')  #1000
    args = parser.parse_args()
    lang = args.lang
    my_arg = args.my_arg
    log_dir = args.log_dir
    train_data = args.train_data
    dev_data = args.dev_data
    model_dir = args.model_dir
    cuda_num = args.cuda_num
    epoch = args.epoch
    print lang, my_arg, log_dir, train_data, dev_data, model_dir, cuda_num, type(cuda_num)

    config = get_conf(lang)  # get_conf(args.lang)
    # config['train_data'] = 'data/bio_cmn_all.txt'
    # config['USE_CUDA'] = False
    # config['batch_size'] = 8

    # my_arg = 0
    config['train_data'] = train_data
    config['dev_data'] = dev_data
    config['model_dir'] = model_dir
    config['cuda_num'] = cuda_num
    config['epoch'] = epoch
    char_emb = LoadEmbedding(config['char_embedding'])
    word_emb = LoadEmbedding(config['embedding_file'])

    # word_emb = LoadEmbedding(config['eval_word_emb'])
    # char_emb = LoadEmbedding(config['eval_char_emb'])

    print 'finish loading embedding'
    encoder = CMNBioCNNEncoder(config, word_emb, char_emb, config['dropout'])
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], config['output_dim'], config['dropout'], config['decoder_layers'])
    # en_dict = torch.load('model/encoder_params.pkl')
    # de_dict = torch.load('model/decoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # encoder.load_state_dict(en_dict)
    # decoder.load_state_dict(de_dict)
    decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])
    train(my_arg, log_dir, config, encoder, decoder, encoder_optimizer, decoder_optimizer)
    torch.save(decoder.state_dict(), os.path.join(config['model_dir'], 'decoder_params.pkl'))
    torch.save(encoder.state_dict(), os.path.join(config['model_dir'], 'encoder_params.pkl'))


