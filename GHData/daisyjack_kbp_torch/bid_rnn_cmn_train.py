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
from bio_model import BioRnnDecoder, CMNBioCNNEncoder, BidRnnBioDecoder
from batch_getter import BatchGetter, get_source_mask, CMNBioBatchGetter
import codecs
import time
import random
import math
from bio_eval_ner import cmn_eval_all, bid_cmn_eval_all
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


def train_iteration(logger, config, my_arg, step,
                    encoder, bidencoder, decoder, encoder_optimizer, bidencoder_optimizer, decoder_optimizer, this_batch):
    # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['hidden_size']))
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    this_batch_num = len(this_batch[2])
    this_batch_max_target = max(this_batch[2])
    last_hidden = Variable(torch.zeros(1, this_batch_num, config['hidden_size']))
    bid_init_hidden = Variable(torch.zeros(config['decoder_layers']*2, this_batch_num, config['hidden_size']))
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
        bid_init_hidden = bid_init_hidden.cuda(config['cuda_num'])

    encoder_outputs = encoder(step, data, h_0, this_batch[3])
    source_mask = Variable(get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['cuda_num'])
    encoder_outputs = encoder_outputs * source_mask
    encoder_outputs = bidencoder(bid_init_hidden, encoder_outputs, this_batch[3])

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
            a = random_pick([0, 1], [rate, 1 - rate])
            if a == 0:
                word_input = target[:, time_step]
            else:
                value, label = label_logits.data.topk(1)
                # decoder_out_label.append(label)
                word_input = Variable(label)  # Chosen word is next input
                if config['USE_CUDA']:
                    word_input = word_input.cuda(config['cuda_num'])

    loss = masked_cross_entropy(seq_label_prob.transpose(0, 1).contiguous(), target, target_length)
    # loss = masked_cross_entropy(F.softmax(decoder_prob.transpose(0,1).contiguous()), target, length)
    print 'loss: ', loss.data[0]
    logger.scalar_summary('loss', loss.data[0], step)
    loss.backward()
    e_before_step = [(tag, to_np(value)) for tag, value in encoder.named_parameters()]
    b_before_step = [(tag, to_np(value)) for tag, value in bidencoder.named_parameters()]
    d_before_step = [(tag, to_np(value)) for tag, value in decoder.named_parameters()]

    clip_grad_norm(decoder.parameters(), config['clip_norm'])
    clip_grad_norm(encoder.parameters(), config['clip_norm'])
    decoder_optimizer.step()
    encoder_optimizer.step()
    bidencoder_optimizer.step()
    e_after_step = [(tag, to_np(value)) for tag, value in encoder.named_parameters()]
    b_after_step = [(tag, to_np(value)) for tag, value in bidencoder.named_parameters()]
    d_after_step = [(tag, to_np(value)) for tag, value in decoder.named_parameters()]
    for before, after in zip(e_before_step, e_after_step):
        if before[0] == after[0]:
            tag = before[0]
            value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
            tag = tag.replace('.', '/')
            if value is not None:
                logger.scalar_summary(tag + '/grad_ratio', value, step)
    for before, after in zip(b_before_step, b_after_step):
        if before[0] == after[0]:
            tag = before[0]
            value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
            tag = tag.replace('.', '/')
            if value is not None:
                logger.scalar_summary(tag + '/grad_ratio', value, step)

    for before, after in zip(d_before_step, d_after_step):
        if before[0] == after[0]:
            tag = before[0]
            value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
            tag = tag.replace('.', '/')
            if value is not None:
                logger.scalar_summary(tag + '/grad_ratio', value, step)




def train_epoch(logger, config, my_arg, epoch, ex_iterations, batch_getter,
                encoder, bidencoder, decoder, encoder_optimizer, bidencoder_optimizer, decoder_optimizer):
    batch_getter.reset()
    for iteration, this_batch in enumerate(batch_getter):
        time0 = time.time()
        print 'epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration)
        train_iteration(logger, config, my_arg, ex_iterations + iteration,
                        encoder, bidencoder, decoder, encoder_optimizer, bidencoder_optimizer, decoder_optimizer, this_batch)
        time1 = time.time()
        print 'this iteration time: ', time1 - time0, '\n'
        if (ex_iterations + iteration) % config['save_freq'] == 0:
            torch.save(decoder.state_dict(), os.path.join(config['model_dir'], 'decoder_params.pkl'))
            torch.save(encoder.state_dict(), os.path.join(config['model_dir'], 'encoder_params.pkl'))
            torch.save(bidencoder.state_dict(), os.path.join(config['model_dir'], 'bidencoder_params.pkl'))
    return ex_iterations + iteration


def train(my_arg, log_dir, config, encoder, bidencoder, decoder, encoder_optimizer, bidencoder_optimizer, decoder_optimizer):
    logger = Logger(log_dir)
    log_file = open(os.path.join(log_dir, 'eval_log'), 'w')
    batch_getter = CMNBioBatchGetter(config, config['train_data'], config['batch_size'], shuffle=True, bio=True)
    print 'finish loading data'
    # batch_getter = BatchGetter('data/train.txt', 8)
    ex_iterations = 0
    f_max = 0
    low_epoch = 0
    for i in range(10000):
        # f, p, r = bid_cmn_eval_all(config, log_dir, False)
        result = train_epoch(logger, config, my_arg, i, ex_iterations, batch_getter,
                             encoder, bidencoder, decoder, encoder_optimizer, bidencoder_optimizer, decoder_optimizer)
        ex_iterations = result + 1
        f, p, r = bid_cmn_eval_all(config, log_dir, False)
        log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(i, f, p, r))
        log_file.flush()
        if f >= f_max:
            f_max = f
            low_epoch = 0
            os.system('cp {} {}'.format(os.path.join(config['model_dir'], 'decoder_params.pkl'),
                                        os.path.join(config['model_dir'], 'early_decoder_params.pkl')))
            os.system('cp {} {}'.format(os.path.join(config['model_dir'], 'bidencoder_params.pkl'),
                                        os.path.join(config['model_dir'], 'early_bidencoder_params.pkl')))
            os.system('cp {} {}'.format(os.path.join(config['model_dir'], 'encoder_params.pkl'),
                                        os.path.join(config['model_dir'], 'early_encoder_params.pkl')))
        else:
            low_epoch += 1
            log_file.write('low' + str(low_epoch) + '\n')
            log_file.flush()
        if low_epoch >= config['early_stop']:
            break
    log_file.close()



if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='eng, cmn, spa')
    parser.add_argument('--my_arg', help='mode', type=int)
    parser.add_argument('--log_dir', help='log file dir')
    args = parser.parse_args()
    my_arg = 0  # args.my_arg
    log_dir = 'bio_bid_cmn_logs'  # args.log_dir  # 'bio_bid_cmn_logs'  #
    lang = args.lang
    print log_dir
    print type(args.lang), args.lang

    print type(my_arg), my_arg


    config = get_conf('cmn')  # get_conf(args.lang)
    config['model_dir'] = 'bid_' + config['model_dir']
    config['decoder_layers'] = 1
    config['batch_size'] = 64
    # config['dropout'] = 0.25
    # config['encoder_filter_num'] = 400
    # config['hidden_size'] = 400
    # config['encoder_outputs_size'] = config['hidden_size']
    # config['USE_CUDA'] = False

    # my_arg = 0

    char_emb = LoadEmbedding(config['char_embedding'])
    word_emb = LoadEmbedding(config['embedding_file'])

    # word_emb = LoadEmbedding(config['eval_word_emb'])
    # char_emb = LoadEmbedding(config['eval_char_emb'])

    print 'finish loading embedding'
    encoder = CMNBioCNNEncoder(config, word_emb, char_emb, config['dropout'])
    bidencoder = BidRnnBioDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], config['dropout'], config['decoder_layers'])

    decoder = BioRnnDecoder(config, config['hidden_size']*2, config['hidden_size'],
                            config['decoder_output_size'], config['output_dim'], config['dropout'],
                            1)
    # en_dict = torch.load('model/encoder_params.pkl')
    # de_dict = torch.load('model/decoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # encoder.load_state_dict(en_dict)
    # decoder.load_state_dict(de_dict)
    decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    bidencoder_optimizer = torch.optim.Adadelta(bidencoder.parameters())
    encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        bidencoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])
    train(my_arg, log_dir, config, encoder, bidencoder, decoder, encoder_optimizer, bidencoder_optimizer, decoder_optimizer)
    torch.save(decoder.state_dict(), os.path.join(config['model_dir'], 'decoder_params.pkl'))
    torch.save(encoder.state_dict(), os.path.join(config['model_dir'], 'encoder_params.pkl'))
    torch.save(bidencoder.state_dict(), os.path.join(config['model_dir'], 'bidencoder_params.pkl'))


