# coding: utf-8

from decoder import BahdanauAttnDecoderRNN, MultiDecoder
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm
from masked_cross_entropy import masked_cross_entropy
from configurations import Vocab
import codecs
from configurations import get_conf
from encoder import LoadEmbedding, CNNEncoder, MultiCNNEncoder
from bio_model import BioRnnDecoder, BioCNNEncoder, CMNBioCNNEncoder, BidRnnBioDecoder
from batch_getter import get_source_mask, BatchGetter, BioBatchGetter, CMNBioBatchGetter
import random
import os

class BoundaryPerformance(object):
    def __init__(self, vocab):
        self._vocab = vocab
        self.reset()

    def reset(self):
        self._hit_num = 0
        self._rec_num = 0
        self._lab_num = 0
        self._unmatch = 0

    def _extract_mentions(self, tokens):
        mentions = []
        pos = 0
        mention_stack = []
        for token in tokens:
            if token == 'X':
                pos += 1
            elif token.startswith('('):
                mention_stack.append((token[1:], pos))
            elif token.startswith(')'):
                mention_type = token[1:]
                is_match = False
                for pre in mention_stack[::-1]:
                    if pre[0] == mention_type:
                        is_match = True
                        mention_stack.remove(pre)
                        mentions.append('_'.join((pre[0], str(pre[1]), str(pos))))
                        break
                if not is_match:
                    self._unmatch += 1
        self._unmatch += len(mention_stack)

        return set(mentions)

    def evaluate(self, i, label, rec, out_stream=None, pr=True):
        label = [self._vocab.getWord(l) for l in label]
        rec = [self._vocab.getWord(r) for r in rec]
        if out_stream is not None:
            label_str = ' '.join(label)
            rec_str = ' '.join(rec)
            out_stream.write('{}|||{}\n'.format(label_str, rec_str))
            out_stream.flush()
        mention_lab = self._extags(label)
        mention_rec = self._extags(rec)
        if pr:
            print i, mention_rec
        mention_hit = mention_lab.intersection(mention_rec)
        self._lab_num += len(mention_lab)
        self._rec_num += len(mention_rec)
        self._hit_num += len(mention_hit)

    def get_performance(self):
        p = float(self._hit_num) / float(self._rec_num) if self._rec_num > 0 else 0.0
        r = float(self._hit_num) / float(self._lab_num) if self._lab_num > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f = f * 100
        print 'label={}, rec={}, hit={}, unmatch_start={}'.format(self._lab_num, self._rec_num, self._hit_num, self._unmatch)
        print 'p={},r={}, f={}'.format(p, r, f)
        return (f, p, r)

    def _extags(self, intags):
        alltags = []
        for line in intags:
            tagCurr = []
            tags = line.strip().split(':')
            for tag in tags:
                if tag == 'O' or tag == '':
                    continue
                tagCurr.append(tag)
            alltags.append(tagCurr)
        outtags= []
        for wpos in range(len(alltags)):
            for (tpos,tag) in enumerate(alltags[wpos]):
                if tag.startswith('B-'):
                    start = wpos
                    end = wpos
                    subtag = tag[2:]
                    itag = 'I-' + subtag
                    for p2 in range(wpos+1, len(alltags)):
                        if (len(alltags[p2])> tpos) and alltags[p2][tpos] == itag:
                            end = p2
                        else:
                            break
                    outtags.append('{} {} {}'.format(start, end, subtag))
        return set(outtags)



def eva_one_sentence(config, encoder, decoder, this_batch):
    this_batch_num = len(this_batch[3])
    this_batch_max_seq = max(this_batch[3])
    last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))

    data = Variable(this_batch[0], volatile=True)
    # target = Variable(this_batch[1], volatile=True)
    length = Variable(torch.LongTensor(this_batch[3]), volatile=True)
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size'] / 2),
                   volatile=True)  # encoder gru initial hidden state

    if config['USE_CUDA']:
        last_hidden = last_hidden.cuda(config['cuda_num'])
        word_input = word_input.cuda(config['cuda_num'])
        data = data.cuda(config['cuda_num'])
        # target = target.cuda(config['cuda_num'])
        length = length.cuda(config['cuda_num'])
        h_0 = h_0.cuda(config['cuda_num'])

    encoder_outputs = encoder(0, data, h_0, this_batch[3])
    # encoder_outputs = encoder_outputs.transpose(1, 2)
    # encoder_outputs = encoder_outputs.transpose(0, 1)
    source_mask = Variable(get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['cuda_num'])
    encoder_outputs = encoder_outputs * source_mask
    # encoder_outputs = encoder_outputs.transpose(0, 1)


    # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['encoder_outputs_size']))
    # last_hidden = Variable(torch.randn(config['decoder_layers'], config['batch_size'], config['hidden_size']))
    # decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'], config['decoder_output_size'], config['decoder_layers'])
    # decoder.load_state_dict(torch.load('net_params.pkl'))
    # optimizer = torch.optim.Adadelta(decoder.parameters())
    # word_input = Variable(torch.LongTensor([[0], [1]]))
    # if config['USE_CUDA']:
    #     encoder_outputs = encoder_outputs.cuda(config['cuda_num'])
    #     last_hidden = last_hidden.cuda(config['cuda_num'])
    #     word_input = word_input.cuda(config['cuda_num'])
    #     decoder.cuda(config['cuda_num'])

    # evaluate
    beam = [{'paths':[[],[]], 'prob':Variable(torch.zeros(this_batch_num, 1)),
             'hidden':Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))},{}]  # beam_size*batch_size*([path],hidden)
    beam = []
    for beam_i in range(config['beam_size']):
        prob_init = Variable(torch.zeros(this_batch_num, 1))
        hidden_init = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
        if config['USE_CUDA']:
            prob_init = prob_init.cuda(config['cuda_num'])
            hidden_init = hidden_init.cuda(config['cuda_num'])
        one_beam = {'paths':[], 'prob': prob_init, 'hidden': hidden_init}
        for batch_i in range(this_batch_num):
            one_beam['paths'].append([0])
        beam.append(one_beam)



    # beam = [{'paths':[] , 'tails':range(output_size) },{'paths':[] , 'tails':range(output_size)}]

    # print beam
    for time_step in range(this_batch_max_seq):
        # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)\

        next_prob = []
        cur_hidden_lst = []
        for i, beam_i in enumerate(beam):
            word_input = Variable(torch.LongTensor(this_batch_num, 1).zero_())
            for batch_i in range(len(beam_i['paths'])):
                word_input[batch_i, 0] = beam_i['paths'][batch_i][-1]
            last_hidden = beam_i['hidden']
            if config['USE_CUDA']:
                word_input = word_input.cuda(config['cuda_num'])
                last_hidden = last_hidden.cuda(config['cuda_num'])
            # word_input: (batch, 1)  last_hidden: (layers * directions, batch, hidden) encoder_outputs: (S, B, hidden)
            # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)
            label_prob, cur_hidden = decoder(0, word_input, last_hidden, encoder_outputs[time_step])
            cur_hidden_lst.append(cur_hidden)
            log_label_prob = F.log_softmax(label_prob)
            next_prob.append(beam_i['prob'].expand_as(log_label_prob) + log_label_prob) # (batch_size, output_size)
        cat = torch.cat(next_prob, 1)  # (batch, outputs_size*beam_size)
        # indices:(batch, beam_size)
        value, indices = cat.topk(config['beam_size'], 1)
        beam_num = indices / int(config['decoder_output_size'])  # (batch, beam_size)
        label_num = indices - beam_num * config['decoder_output_size']  # (batch, beam_size)
        beam_num = beam_num.cpu().data
        label_num = label_num.cpu().data
        new_beam = []
        for beam_i in range(config['beam_size']):
            prob_init = Variable(torch.zeros(this_batch_num, 1))
            hidden_init = Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))
            if config['USE_CUDA']:
                prob_init = prob_init.cuda(config['cuda_num'])
                hidden_init = hidden_init.cuda(config['cuda_num'])
            one_beam = {'paths': [], 'prob': prob_init, 'hidden': hidden_init}
            new_beam.append(one_beam)
        for i_batch in range(this_batch_num):
            for i_beam in range(config['beam_size']):
                this_beam_num = beam_num[i_batch][i_beam]
                this_label = label_num[i_batch][i_beam]
                a = beam[this_beam_num]['paths'][i_batch][:]
                a.append(this_label)
                new_beam[i_beam]['paths'].append(a)
                new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
                new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, this_label]

        beam = new_beam
        top_path = beam[0]['paths'][0]
        # if top_path[-1] != config['X']:
        #     print '-----------------------\n', top_path[-1], '\n', '--------------'
        # if top_path.count(config['X']) == this_batch_max_seq and top_path[-1] == config['EOS_token']:
        #     break
        #
        # if top_path.count(config['X']) > this_batch_max_seq:
        #     top_path[-1] = config['EOS_token']
        #     break

    # print beam[0]['paths']  # (output_size, B, 1)
    return top_path


def bid_eval_one_sen(config, encoder, bidencoder, decoder, this_batch):
    this_batch_num = len(this_batch[3])
    this_batch_max_seq = max(this_batch[3])
    last_hidden = Variable(torch.zeros(1, this_batch_num, config['hidden_size']))
    bid_init_hidden = Variable(torch.zeros(config['decoder_layers'] * 2, this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))

    data = Variable(this_batch[0], volatile=True)
    # target = Variable(this_batch[1], volatile=True)
    length = Variable(torch.LongTensor(this_batch[3]), volatile=True)
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size'] / 2),
                   volatile=True)  # encoder gru initial hidden state

    if config['USE_CUDA']:
        last_hidden = last_hidden.cuda(config['cuda_num'])
        word_input = word_input.cuda(config['cuda_num'])
        data = data.cuda(config['cuda_num'])
        # target = target.cuda(config['cuda_num'])
        length = length.cuda(config['cuda_num'])
        h_0 = h_0.cuda(config['cuda_num'])
        bid_init_hidden = bid_init_hidden.cuda(config['cuda_num'])

    encoder_outputs = encoder(0, data, h_0, this_batch[3])
    # encoder_outputs = encoder_outputs.transpose(1, 2)
    # encoder_outputs = encoder_outputs.transpose(0, 1)
    source_mask = Variable(
        get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['cuda_num'])
    encoder_outputs = encoder_outputs * source_mask
    encoder_outputs = bidencoder(bid_init_hidden, encoder_outputs, this_batch[3])

    # evaluate
    beam = [{'paths': [[], []], 'prob': Variable(torch.zeros(this_batch_num, 1)),
             'hidden': Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))},
            {}]  # beam_size*batch_size*([path],hidden)
    beam = []
    for beam_i in range(config['beam_size']):
        prob_init = Variable(torch.zeros(this_batch_num, 1))
        hidden_init = Variable(torch.zeros(1, this_batch_num, config['hidden_size']))
        if config['USE_CUDA']:
            prob_init = prob_init.cuda(config['cuda_num'])
            hidden_init = hidden_init.cuda(config['cuda_num'])
        one_beam = {'paths': [], 'prob': prob_init, 'hidden': hidden_init}
        for batch_i in range(this_batch_num):
            one_beam['paths'].append([0])
        beam.append(one_beam)

    # beam = [{'paths':[] , 'tails':range(output_size) },{'paths':[] , 'tails':range(output_size)}]

    # print beam
    for time_step in range(this_batch_max_seq):
        # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)\

        next_prob = []
        cur_hidden_lst = []
        for i, beam_i in enumerate(beam):
            word_input = Variable(torch.LongTensor(this_batch_num, 1).zero_())
            for batch_i in range(len(beam_i['paths'])):
                word_input[batch_i, 0] = beam_i['paths'][batch_i][-1]
            last_hidden = beam_i['hidden']
            if config['USE_CUDA']:
                word_input = word_input.cuda(config['cuda_num'])
                last_hidden = last_hidden.cuda(config['cuda_num'])
            label_prob, cur_hidden = decoder(0, word_input, last_hidden, encoder_outputs[time_step])
            cur_hidden_lst.append(cur_hidden)
            log_label_prob = F.log_softmax(label_prob)
            next_prob.append(beam_i['prob'].expand_as(log_label_prob) + log_label_prob)  # (batch_size, output_size)
        cat = torch.cat(next_prob, 1)  # (batch, outputs_size*beam_size)
        # indices:(batch, beam_size)
        value, indices = cat.topk(config['beam_size'], 1)
        beam_num = indices / int(config['decoder_output_size'])  # (batch, beam_size)
        label_num = indices - beam_num * config['decoder_output_size']  # (batch, beam_size)
        beam_num = beam_num.cpu().data
        label_num = label_num.cpu().data
        new_beam = []
        for beam_i in range(config['beam_size']):
            prob_init = Variable(torch.zeros(this_batch_num, 1))
            hidden_init = Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))
            if config['USE_CUDA']:
                prob_init = prob_init.cuda(config['cuda_num'])
                hidden_init = hidden_init.cuda(config['cuda_num'])
            one_beam = {'paths': [], 'prob': prob_init, 'hidden': hidden_init}
            new_beam.append(one_beam)
        for i_batch in range(this_batch_num):
            for i_beam in range(config['beam_size']):
                this_beam_num = beam_num[i_batch][i_beam]
                this_label = label_num[i_batch][i_beam]
                a = beam[this_beam_num]['paths'][i_batch][:]
                a.append(this_label)
                new_beam[i_beam]['paths'].append(a)
                new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
                new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, this_label]

        beam = new_beam
        top_path = beam[0]['paths'][0]
    return top_path





def eva_one_sentence_vib(config, encoder, decoder, this_batch):
    this_batch_num = len(this_batch[3])
    this_batch_max_seq = max(this_batch[3])
    last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))

    data = Variable(this_batch[0], volatile=True)
    target = Variable(this_batch[1], volatile=True)
    length = Variable(torch.LongTensor(this_batch[3]), volatile=True)
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size'] / 2), volatile=True)  # encoder gru initial hidden state

    if config['USE_CUDA']:
        last_hidden = last_hidden.cuda(config['cuda_num'])
        word_input = word_input.cuda(config['cuda_num'])
        data = data.cuda(config['cuda_num'])
        target = target.cuda(config['cuda_num'])
        length = length.cuda(config['cuda_num'])
        h_0 = h_0.cuda(config['cuda_num'])

    encoder_outputs = encoder(0, data, h_0, this_batch[3])
    # encoder_outputs = encoder_outputs.transpose(1, 2)
    # encoder_outputs = encoder_outputs.transpose(0, 1)
    source_mask = Variable(get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['cuda_num'])
    encoder_outputs = encoder_outputs * source_mask


    # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['encoder_outputs_size']))
    # last_hidden = Variable(torch.randn(config['decoder_layers'], config['batch_size'], config['hidden_size']))
    # decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'], config['decoder_output_size'], config['decoder_layers'])
    # decoder.load_state_dict(torch.load('net_params.pkl'))
    # optimizer = torch.optim.Adadelta(decoder.parameters())
    # word_input = Variable(torch.LongTensor([[0], [1]]))
    # if config['USE_CUDA']:
    #     encoder_outputs = encoder_outputs.cuda(config['cuda_num'])
    #     last_hidden = last_hidden.cuda(config['cuda_num'])
    #     word_input = word_input.cuda(config['cuda_num'])
    #     decoder.cuda(config['cuda_num'])

    # evaluate
    beam = [{'paths':[[],[]], 'prob':Variable(torch.zeros(this_batch_num, 1)),
             'hidden':Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))},{}]  # beam_size*batch_size*([path],hidden)
    beam = []
    tag_size = 18
    for beam_i in range(tag_size):
        prob_init = Variable(torch.zeros(this_batch_num, 1))
        hidden_init = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
        if config['USE_CUDA']:
            prob_init = prob_init.cuda(config['cuda_num'])
            hidden_init = hidden_init.cuda(config['cuda_num'])
        one_beam = {'paths':[], 'prob': prob_init, 'hidden': hidden_init}
        for batch_i in range(this_batch_num):
            one_beam['paths'].append([0])
        beam.append(one_beam)



    # beam = [{'paths':[] , 'tails':range(output_size) },{'paths':[] , 'tails':range(output_size)}]

    # print beam
    for time_step in range(this_batch_max_seq*3):
        # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)\

        next_prob = []
        cur_hidden_lst = []
        for i, beam_i in enumerate(beam):
            word_input = Variable(torch.LongTensor(this_batch_num, 1).zero_())
            for batch_i in range(len(beam_i['paths'])):
                word_input[batch_i, 0] = beam_i['paths'][batch_i][-1]
            last_hidden = beam_i['hidden']
            if config['USE_CUDA']:
                word_input = word_input.cuda(config['cuda_num'])
                last_hidden = last_hidden.cuda(config['cuda_num'])
            # word_input: (batch, 1)  last_hidden: (layers * directions, batch, hidden) encoder_outputs: (S, B, hidden)
            # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)
            label_prob, cur_hidden, attn_weights = decoder(0, word_input, last_hidden, encoder_outputs)
            cur_hidden_lst.append(cur_hidden)
            log_label_prob = F.log_softmax(label_prob)
            next_prob.append(beam_i['prob'].expand_as(log_label_prob) + log_label_prob) # (batch_size, output_size)
        # cat = torch.cat(next_prob, 1)  # (batch, outputs_size*beam_size)
        batch_best_indices = []
        for batch_i in range(this_batch_num):
            cat = [prob[batch_i, :].unsqueeze(0) for prob in next_prob]
            cat = torch.cat(cat, 0)
            values, indices = cat.topk(1, 0)  # indices: (1, tag_size)
            batch_best_indices.append(indices.data)

        new_beam = []
        for beam_i in range(tag_size):
            prob_init = Variable(torch.zeros(this_batch_num, 1))
            hidden_init = Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))
            if config['USE_CUDA']:
                prob_init = prob_init.cuda(config['cuda_num'])
                hidden_init = hidden_init.cuda(config['cuda_num'])
            one_beam = {'paths': [], 'prob': prob_init, 'hidden': hidden_init}
            new_beam.append(one_beam)
        for i_batch in range(this_batch_num):
            for i_beam in range(tag_size):
                this_beam_num = batch_best_indices[i_batch][0, i_beam]
                a = beam[this_beam_num]['paths'][i_batch][:]
                a.append(i_beam)
                new_beam[i_beam]['paths'].append(a)
                new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
                new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, i_beam]

        beam = new_beam
        batch_best_path = []
        for i_batch in range(this_batch_num):
            this_batch_best_path = beam[0]['paths'][i_batch]
            this_batch_best_prob = beam[0]['prob'][i_batch, 0]
            for i_beam in range(tag_size):
                if beam[i_beam]['prob'][i_batch, 0] > this_batch_best_prob:
                    this_batch_best_path = beam[i_beam]['paths'][i_batch]
            batch_best_path.append(this_batch_best_path)

        top_path = batch_best_path[0]
        if top_path.count(config['X']) == this_batch_max_seq and top_path[-1] == config['EOS_token']:
            break

        if top_path.count(config['X']) > this_batch_max_seq:
            top_path[-1] = config['EOS_token']
            break

    # print beam[0]['paths']  # (output_size, B, 1)
    return top_path



def pad_seq(seq, max_length):
    seq += [config['PAD_token'] for i in range(max_length - len(seq))]
    return seq



def evaluate_all(config, my_arg, log_dir, pr=True):
    emb = LoadEmbedding(config['eval_emb'])
    print 'finish loading embedding'
    encoder = BioCNNEncoder(config, emb, dropout_p=0)
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_getter = BioBatchGetter(config, config['dev_data'], 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = config['BioOutTags']
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_path = eva_one_sentence(config, encoder, decoder, this_batch)
        top_path = top_path[1:]
        # print [ner_tag.getWord(tag) for tag in top_path]
        evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
        if i % 100 == 0:
            print '{} sentences processed'.format(i)
            evaluator.get_performance()

    return evaluator.get_performance()

def batch_evaluate_all(config, my_arg, log_dir, pr=True):
    emb = LoadEmbedding(config['eval_emb'])
    print 'finish loading embedding'
    encoder = BioCNNEncoder(config, emb, dropout_p=0)
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_num = 50
    batch_getter = BioBatchGetter(config, config['dev_data'], batch_num, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = config['BioOutTags']
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_paths = batch_eva_one_sentence(config, encoder, decoder, this_batch)
        for batch_no, top_path in enumerate(top_paths):
            this_batch_len = this_batch[3][batch_no]
            top_path = top_path[1:this_batch_len + 1]
            # print [ner_tag.getWord(tag) for tag in top_path]
            evaluator.evaluate(i, this_batch[1].numpy()[batch_no, :].tolist()[:this_batch_len], top_path, out_file, pr)
        print '{} sentences processed'.format((i + 1) * batch_num)
        evaluator.get_performance()

    return evaluator.get_performance()


def bid_eval_all(config, my_arg, log_dir, pr=True):
    emb = LoadEmbedding(config['eval_emb'])
    print 'finish loading embedding'
    encoder = BioCNNEncoder(config, emb, dropout_p=0)
    decoder = BidRnnBioDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_getter = BioBatchGetter(config, config['dev_data'], 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = config['BioOutTags']
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_path = bid_eval_one_sen(config, encoder, decoder, this_batch)
        # print [ner_tag.getWord(tag) for tag in top_path]
        evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path.numpy().tolist(), out_file, pr)
        if i % 100 == 0:
            print '{} sentences processed'.format(i)
            evaluator.get_performance()

    return evaluator.get_performance()

def bid_cmn_eval_all(config, log_dir, pr=True):
    word_emb = LoadEmbedding(config['eval_word_emb'])
    char_emb = LoadEmbedding(config['eval_char_emb'])
    print 'finish loading embedding'
    encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
    bidencoder = BidRnnBioDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                  config['decoder_output_size'], 0, config['decoder_layers'])
    decoder = BioRnnDecoder(config, config['hidden_size'] * 2, config['hidden_size'],
                            config['decoder_output_size'], config['output_dim'], 0,
                            1)
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    bid_dict = torch.load(os.path.join(config['model_dir'], 'bidencoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    bidencoder.load_state_dict(bid_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_getter = CMNBioBatchGetter(config, config['dev_data'], 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        bidencoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = config['BioOutTags']
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_path = bid_eval_one_sen(config, encoder, bidencoder, decoder, this_batch)
        top_path = top_path[1:]
        # print [ner_tag.getWord(tag) for tag in top_path]
        evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
        if i % 100 == 0:
            print '{} sentences processed'.format(i)
            evaluator.get_performance()

    return evaluator.get_performance()

def cmn_eval_all(config, log_dir, pr=True):
    word_emb = LoadEmbedding(config['eval_word_emb'])
    char_emb = LoadEmbedding(config['eval_char_emb'])
    print 'finish loading embedding'
    encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_getter = CMNBioBatchGetter(config, config['dev_data'], 1, shuffle=False)
    # batch_getter = CMNBioBatchGetter(config, 'data/cmn.txt', 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = config['BioOutTags']
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_path = eva_one_sentence(config, encoder, decoder, this_batch)
        top_path = top_path[1:]
        # print [ner_tag.getWord(tag) for tag in top_path]
        evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
        if i % 100 == 0:
            print '{} sentences processed'.format(i)
            evaluator.get_performance()

    return evaluator.get_performance()

def batch_cmn_eval_all(config, log_dir, pr=True):
    word_emb = LoadEmbedding(config['eval_word_emb'])
    char_emb = LoadEmbedding(config['eval_char_emb'])
    print 'finish loading embedding'
    encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                     config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_num = 50
    batch_getter = CMNBioBatchGetter(config, config['dev_data'], batch_num, shuffle=False)
    # batch_getter = CMNBioBatchGetter(config, 'data/cmn.txt', 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = config['BioOutTags']
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_paths = batch_eva_one_sentence(config, encoder, decoder, this_batch)
        for batch_no, top_path in enumerate(top_paths):
            this_batch_len = this_batch[3][batch_no]
            top_path = top_path[1:this_batch_len+1]
            # print [ner_tag.getWord(tag) for tag in top_path]
            evaluator.evaluate(i, this_batch[1].numpy()[batch_no, :].tolist()[:this_batch_len], top_path, out_file, pr)
        print '{} sentences processed'.format((i + 1) * batch_num)
        evaluator.get_performance()

    return evaluator.get_performance()


def cmn_infer_eval(config, data):
    # config['USE_CUDA'] = False
    word_emb = LoadEmbedding(config['eval_word_emb'])
    char_emb = LoadEmbedding(config['eval_char_emb'])
    print 'finish loading embedding'
    encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                            config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'decoder_params.pkl'))
    # en_dict = torch.load('bio_model_eng/early_encoder_params.pkl')
    # de_dict = torch.load('bio_model_eng/early_decoder_params.pkl')
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    # batch_getter = BioBatchGetter(config, config['dev_data'], 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    # ner_tag = config['BioOutTags']
    # evaluator = BoundaryPerformance(ner_tag)
    # evaluator.reset()
    #
    # out_file = codecs.open('data/bio_eva_result', mode='wb', encoding='utf-8')

    result = eva_one_sentence(config, encoder, decoder, data)
    return result


def infer_eval(config, data):
    emb = LoadEmbedding(config['eval_emb'])
    # emb = config['loaded_emb']
    print 'finish loading embedding'
    encoder = BioCNNEncoder(config, emb, dropout_p=0)
    decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                            config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
    en_dict = torch.load(os.path.join(config['model_dir'], 'early_encoder_params.pkl'))
    de_dict = torch.load(os.path.join(config['model_dir'], 'early_decoder_params.pkl'))
    # en_dict = torch.load('bio_model_eng/early_encoder_params.pkl')
    # de_dict = torch.load('bio_model_eng/early_decoder_params.pkl')
    # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    # batch_getter = BioBatchGetter(config, config['dev_data'], 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    # ner_tag = config['BioOutTags']
    # evaluator = BoundaryPerformance(ner_tag)
    # evaluator.reset()
    #
    # out_file = codecs.open('data/bio_eva_result', mode='wb', encoding='utf-8')

    result = eva_one_sentence(config, encoder, decoder, data)
    return result

    # for i, this_batch in enumerate(batch_getter):
    #     top_path = eva_one_sentence(config, encoder, decoder, this_batch)
    #     top_path = top_path[1:]
    #     # print [ner_tag.getWord(tag) for tag in top_path]
    #     evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
    #     if i % 100 == 0:
    #         print '{} sentences processed'.format(i)
    #         evaluator.get_performance()
    #
    # return evaluator.get_performance()

def batch_eva_one_sentence(config, encoder, decoder, this_batch):
    this_batch_num = len(this_batch[3])
    this_batch_max_seq = max(this_batch[3])
    last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))

    data = Variable(this_batch[0], volatile=True)
    # target = Variable(this_batch[1], volatile=True)
    length = Variable(torch.LongTensor(this_batch[3]), volatile=True)
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size'] / 2),
                   volatile=True)  # encoder gru initial hidden state

    if config['USE_CUDA']:
        last_hidden = last_hidden.cuda(config['cuda_num'])
        word_input = word_input.cuda(config['cuda_num'])
        data = data.cuda(config['cuda_num'])
        # target = target.cuda(config['cuda_num'])
        length = length.cuda(config['cuda_num'])
        h_0 = h_0.cuda(config['cuda_num'])

    encoder_outputs = encoder(0, data, h_0, this_batch[3])
    # encoder_outputs = encoder_outputs.transpose(1, 2)
    # encoder_outputs = encoder_outputs.transpose(0, 1)
    source_mask = Variable(get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['cuda_num'])
    encoder_outputs = encoder_outputs * source_mask
    # encoder_outputs = encoder_outputs.transpose(0, 1)


    # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['encoder_outputs_size']))
    # last_hidden = Variable(torch.randn(config['decoder_layers'], config['batch_size'], config['hidden_size']))
    # decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'], config['decoder_output_size'], config['decoder_layers'])
    # decoder.load_state_dict(torch.load('net_params.pkl'))
    # optimizer = torch.optim.Adadelta(decoder.parameters())
    # word_input = Variable(torch.LongTensor([[0], [1]]))
    # if config['USE_CUDA']:
    #     encoder_outputs = encoder_outputs.cuda(config['cuda_num'])
    #     last_hidden = last_hidden.cuda(config['cuda_num'])
    #     word_input = word_input.cuda(config['cuda_num'])
    #     decoder.cuda(config['cuda_num'])

    # evaluate
    beam = [{'paths':[[],[]], 'prob':Variable(torch.zeros(this_batch_num, 1)),
             'hidden':Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))},{}]  # beam_size*batch_size*([path],hidden)
    beam = []
    for beam_i in range(config['beam_size']):
        prob_init = Variable(torch.zeros(this_batch_num, 1))
        hidden_init = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
        if config['USE_CUDA']:
            prob_init = prob_init.cuda(config['cuda_num'])
            hidden_init = hidden_init.cuda(config['cuda_num'])
        one_beam = {'paths':[], 'prob': prob_init, 'hidden': hidden_init}
        for batch_i in range(this_batch_num):
            one_beam['paths'].append([0])
        beam.append(one_beam)



    # beam = [{'paths':[] , 'tails':range(output_size) },{'paths':[] , 'tails':range(output_size)}]

    # print beam
    for time_step in range(this_batch_max_seq):
        # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)\

        next_prob = []
        cur_hidden_lst = []
        for i, beam_i in enumerate(beam):
            word_input = Variable(torch.LongTensor(this_batch_num, 1).zero_())
            for batch_i in range(len(beam_i['paths'])):
                word_input[batch_i, 0] = beam_i['paths'][batch_i][-1]
            last_hidden = beam_i['hidden']
            if config['USE_CUDA']:
                word_input = word_input.cuda(config['cuda_num'])
                last_hidden = last_hidden.cuda(config['cuda_num'])
            # word_input: (batch, 1)  last_hidden: (layers * directions, batch, hidden) encoder_outputs: (S, B, hidden)
            # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)
            label_prob, cur_hidden = decoder(0, word_input, last_hidden, encoder_outputs[time_step])
            cur_hidden_lst.append(cur_hidden)
            log_label_prob = F.log_softmax(label_prob)
            next_prob.append(beam_i['prob'].expand_as(log_label_prob) + log_label_prob) # (batch_size, output_size)
        cat = torch.cat(next_prob, 1)  # (batch, outputs_size*beam_size)
        # indices:(batch, beam_size)
        value, indices = cat.topk(config['beam_size'], 1)
        beam_num = indices / int(config['decoder_output_size'])  # (batch, beam_size)
        label_num = indices - beam_num * config['decoder_output_size']  # (batch, beam_size)
        beam_num = beam_num.cpu().data
        label_num = label_num.cpu().data
        new_beam = []
        for beam_i in range(config['beam_size']):
            prob_init = Variable(torch.zeros(this_batch_num, 1))
            hidden_init = Variable(torch.randn(config['decoder_layers'], this_batch_num, config['hidden_size']))
            if config['USE_CUDA']:
                prob_init = prob_init.cuda(config['cuda_num'])
                hidden_init = hidden_init.cuda(config['cuda_num'])
            one_beam = {'paths': [], 'prob': prob_init, 'hidden': hidden_init}
            new_beam.append(one_beam)
        for i_batch in range(this_batch_num):
            for i_beam in range(config['beam_size']):
                this_beam_num = beam_num[i_batch][i_beam]
                this_label = label_num[i_batch][i_beam]
                a = beam[this_beam_num]['paths'][i_batch][:]

                if time_step > this_batch[3][i_batch]-1:
                    new_beam[i_beam]['paths'].append(beam[i_beam]['paths'][i_batch][:])
                    new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
                    new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, this_label]
                else:
                    a.append(this_label)
                    new_beam[i_beam]['paths'].append(a)
                    new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
                    new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, this_label]
                # a.append(this_label)
                # new_beam[i_beam]['paths'].append(a)
                # new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
                # new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, this_label]

        beam = new_beam
    top_path = beam[0]['paths']
        # if top_path[-1] != config['X']:
        #     print '-----------------------\n', top_path[-1], '\n', '--------------'
        # if top_path.count(config['X']) == this_batch_max_seq and top_path[-1] == config['EOS_token']:
        #     break
        #
        # if top_path.count(config['X']) > this_batch_max_seq:
        #     top_path[-1] = config['EOS_token']
        #     break

    # print beam[0]['paths']  # (output_size, B, 1)
    return top_path



if __name__ == '__main__':
    def free_batch_evaluate_all(config, my_arg, log_dir, pr=True):
        emb = LoadEmbedding(config['eval_emb'])
        print 'finish loading embedding'
        encoder = BioCNNEncoder(config, emb, dropout_p=0)
        decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
        en_dict = torch.load(os.path.join(config['model_dir'], 'early_encoder_params.pkl'))
        de_dict = torch.load(os.path.join(config['model_dir'], 'early_decoder_params.pkl'))
        # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
        # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
        # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
        # print en_dict.keys()
        encoder.load_state_dict(en_dict)
        decoder.load_state_dict(de_dict)
        # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
        # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
        # decoder_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()
        batch_num = 50
        batch_getter = BioBatchGetter(config, config['dev_data'], batch_num, shuffle=False)
        if config['USE_CUDA']:
            encoder.cuda(config['cuda_num'])
            decoder.cuda(config['cuda_num'])

        ner_tag = config['BioOutTags']
        evaluator = BoundaryPerformance(ner_tag)
        evaluator.reset()

        out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

        for i, this_batch in enumerate(batch_getter):
            top_paths = batch_eva_one_sentence(config, encoder, decoder, this_batch)
            for batch_no, top_path in enumerate(top_paths):
                this_batch_len = this_batch[3][batch_no]
                top_path = top_path[1:this_batch_len + 1]
                # print [ner_tag.getWord(tag) for tag in top_path]
                evaluator.evaluate(i, this_batch[1].numpy()[batch_no, :].tolist()[:this_batch_len], top_path, out_file,
                                   pr)
            print '{} sentences processed'.format((i + 1) * batch_num)
            evaluator.get_performance()

        return evaluator.get_performance()


    def free_evaluate_all(config, my_arg, log_dir, pr=True):
        emb = LoadEmbedding(config['eval_emb'])
        print 'finish loading embedding'
        encoder = BioCNNEncoder(config, emb, dropout_p=0)
        decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
        en_dict = torch.load(os.path.join(config['model_dir'], 'early_encoder_params.pkl'))
        de_dict = torch.load(os.path.join(config['model_dir'], 'early_decoder_params.pkl'))
        # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
        # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
        # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
        # print en_dict.keys()
        encoder.load_state_dict(en_dict)
        decoder.load_state_dict(de_dict)
        # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
        # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
        # decoder_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()
        batch_getter = BioBatchGetter(config, config['dev_data'], 1, shuffle=False)
        if config['USE_CUDA']:
            encoder.cuda(config['cuda_num'])
            decoder.cuda(config['cuda_num'])

        ner_tag = config['BioOutTags']
        evaluator = BoundaryPerformance(ner_tag)
        evaluator.reset()

        out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

        for i, this_batch in enumerate(batch_getter):
            top_path = eva_one_sentence(config, encoder, decoder, this_batch)
            top_path = top_path[1:]
            # print [ner_tag.getWord(tag) for tag in top_path]
            evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
            if i % 100 == 0:
                print '{} sentences processed'.format(i)
                evaluator.get_performance()

        return evaluator.get_performance()


    def cmn_eval_free(config, log_dir, pr=True):
        word_emb = LoadEmbedding(config['eval_word_emb'])
        char_emb = LoadEmbedding(config['eval_char_emb'])
        print 'finish loading embedding'
        encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
        decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
        en_dict = torch.load(os.path.join(config['model_dir'], 'early_encoder_params.pkl'))
        de_dict = torch.load(os.path.join(config['model_dir'], 'early_decoder_params.pkl'))
        # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
        # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
        # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
        # print en_dict.keys()
        encoder.load_state_dict(en_dict)
        decoder.load_state_dict(de_dict)
        # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
        # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
        # decoder_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()
        batch_getter = CMNBioBatchGetter(config, config['dev_data'], 1, shuffle=False)
        # batch_getter = CMNBioBatchGetter(config, 'data/cmn.txt', 1, shuffle=False)
        if config['USE_CUDA']:
            encoder.cuda(config['cuda_num'])
            decoder.cuda(config['cuda_num'])

        ner_tag = config['BioOutTags']
        evaluator = BoundaryPerformance(ner_tag)
        evaluator.reset()

        out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

        for i, this_batch in enumerate(batch_getter):
            top_path = eva_one_sentence(config, encoder, decoder, this_batch)
            top_path = top_path[1:]
            # print [ner_tag.getWord(tag) for tag in top_path]
            evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
            if (i+1) % 100 == 0:
                print '{} sentences processed'.format(i+1)
                evaluator.get_performance()

        return evaluator.get_performance()


    def free_batch_cmn_eval_all(config, log_dir, pr=True):
        word_emb = LoadEmbedding(config['eval_word_emb'])
        char_emb = LoadEmbedding(config['eval_char_emb'])
        print 'finish loading embedding'
        encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
        decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
        en_dict = torch.load(os.path.join(config['model_dir'], 'early_encoder_params.pkl'))
        de_dict = torch.load(os.path.join(config['model_dir'], 'early_decoder_params.pkl'))
        # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
        # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
        # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
        # print en_dict.keys()
        encoder.load_state_dict(en_dict)
        decoder.load_state_dict(de_dict)
        # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
        # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
        # decoder_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()
        batch_num = 100
        batch_getter = CMNBioBatchGetter(config, config['dev_data'], batch_num, shuffle=False)
        # batch_getter = CMNBioBatchGetter(config, 'data/cmn.txt', 1, shuffle=False)
        if config['USE_CUDA']:
            encoder.cuda(config['cuda_num'])
            decoder.cuda(config['cuda_num'])

        ner_tag = config['BioOutTags']
        evaluator = BoundaryPerformance(ner_tag)
        evaluator.reset()

        out_file = codecs.open(os.path.join(log_dir, 'bio_eva_result'), mode='wb', encoding='utf-8')

        for i, this_batch in enumerate(batch_getter):
            top_paths = batch_eva_one_sentence(config, encoder, decoder, this_batch)
            for batch_no, top_path in enumerate(top_paths):
                this_batch_len = this_batch[3][batch_no]
                top_path = top_path[1:this_batch_len + 1]
                # print [ner_tag.getWord(tag) for tag in top_path]
                evaluator.evaluate(i, this_batch[1].numpy()[batch_no, :].tolist()[:this_batch_len], top_path, out_file,
                                   pr)
            print '{} sentences processed'.format((i + 1) * batch_num)
            evaluator.get_performance()

        return evaluator.get_performance()


    # config = get_conf('eng')
    # config['dev_data'] = 'data/eng_data/eng_0.3eval.txt'#'data/spa_test_eval0.2.txt'
    # # config['beam_size'] = 24
    # config['model_dir'] = 'eng_bio_model1' #''spa_bio_model0'
    # config['cuda_num'] = 0
    # # evaluate_free(config, 0, 'test', False)
    # free_batch_evaluate_all(config, 0, 'test', False)
    # free_evaluate_all(config, 0, 'test', False)

    config = get_conf('cmn')
    config['dev_data'] = 'data/cmn_data/cmn_0.3eval.txt'
    config['model_dir'] = 'cmn_bio_model1'
    # cmn_eval_free(config, 'test', False)
    free_batch_cmn_eval_all(config, 'test', False)



    # batch_getter = BatchGetter('data/dev')
    # this_batch = next(batch_getter)
    # print this_batch[1].numpy()[0, :].tolist()
    # print _extags(['O', 'O', 'O', 'B-GPE_NAM', 'I-GPE_NAM', 'B-PER_NAM', 'O'])






