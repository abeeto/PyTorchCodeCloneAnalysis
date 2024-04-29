# coding: utf-8

from decoder import BahdanauAttnDecoderRNN, MultiDecoder
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm
from masked_cross_entropy import masked_cross_entropy
from configurations import config, Vocab
import codecs
from encoder import LoadEmbedding, CNNEncoder, MultiCNNEncoder
from batch_getter import get_source_mask, BatchGetter
import random

# def train_iteration(encoder, decoder, encoder_optimizer, decoder_optimizer, this_batch):
#     # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['hidden_size']))
#     this_batch_num = len(this_batch[2])
#     this_batch_max_seq = max(this_batch[2])
#     last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
#     word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))
#
#     data = Variable(this_batch[0])
#     target = Variable(this_batch[1])
#     length = Variable(torch.LongTensor(this_batch[2]))
#
#     if config['USE_CUDA']:
#         last_hidden = last_hidden.cuda(config['cuda_num'])
#         word_input = word_input.cuda(config['cuda_num'])
#         data = data.cuda(config['cuda_num'])
#         target = target.cuda(config['cuda_num'])
#         length = length.cuda(config['cuda_num'])
#
#     encoder_outputs = encoder(data)
#     encoder_outputs = encoder_outputs.transpose(1,2)
#     encoder_outputs = encoder_outputs.transpose(0,1)
#
#
#
#     # decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'], config['decoder_output_size'], config['decoder_layers'])
#     # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
#     # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
#     # word_input = Variable(torch.LongTensor([[0], [1]]))
#     # target = Variable(torch.LongTensor([[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]]))  # (batch, max_label_length)
#
#     # length = Variable(torch.LongTensor([5,7]))
#
#         # decoder.cuda(config['cuda_num'])
#
#
#     # train
#     decoder_out_label = []
#     seq_label_prob = Variable(torch.zeros(this_batch_max_seq, this_batch_num, config['decoder_output_size']))
#     if config['USE_CUDA']:
#         seq_label_prob = seq_label_prob.cuda()
#     for time_step in range(this_batch_max_seq):
#         label_prob, cur_hidden, attn_weights = decoder(word_input, last_hidden, encoder_outputs)
#         last_hidden = cur_hidden
#         seq_label_prob[time_step] = label_prob
#         # Choose top word from label_prob
#         value, label = label_prob.topk(1)
#         decoder_out_label.append(label.data)
#         # not teacher-forcing
#         # word_input = label
#
#         # teacher-forcing
#         word_input = target[:, time_step]
#
#     print max(this_batch[2]) # (output_size, B, 1)
#     decoder_prob = Variable(torch.FloatTensor([[[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]], [[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]]]))
#
#     if config['USE_CUDA']:
#         decoder_prob = decoder_prob.cuda(config['cuda_num'])
#
#     loss = masked_cross_entropy(F.softmax(seq_label_prob.transpose(0,1).contiguous()), target, length)
#     # loss = masked_cross_entropy(F.softmax(decoder_prob.transpose(0,1).contiguous()), target, length)
#     print loss
#     decoder_optimizer.zero_grad()
#     encoder_optimizer.zero_grad()
#     loss.backward()
#     clip = clip_grad_norm(decoder.parameters(), config['clip_norm'])
#     e_clip = clip_grad_norm(encoder.parameters(), config['clip_norm'])
#     decoder_optimizer.step()
#     encoder_optimizer.step()
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

# def train_epoch(epoch, batch_getter, encoder, decoder, encoder_optimizer, decoder_optimizer):
#     batch_getter.reset()
#     for iteration, this_batch in enumerate(batch_getter):
#         print 'epoch: {}, iteraton: {}'.format(epoch, iteration)
#         train_iteration(encoder, decoder, encoder_optimizer, decoder_optimizer, this_batch)
#
# def train(encoder, decoder, encoder_optimizer, decoder_optimizer):
#     batch_getter = BatchGetter('data/train')
#     for i in range(100):
#         train_epoch(i, batch_getter, encoder, decoder, encoder_optimizer, decoder_optimizer)


# if __name__ == '__main__':
#     emb = LoadEmbedding('res/embedding.txt')
#     print 'finish loading embedding'
#     encoder = CNNEncoder(emb)
#     decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'],
#                                      config['decoder_output_size'], config['decoder_layers'])
#     decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
#     encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
#     if config['USE_CUDA']:
#         encoder.cuda(config['cuda_num'])
#         decoder.cuda(config['cuda_num'])
#     train(encoder, decoder, encoder_optimizer, decoder_optimizer)
#     torch.save(decoder.state_dict(), 'decoder_params.pkl')
#     torch.save(encoder.state_dict(), 'encoder_params.pkl')



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
        mention_lab = self._extract_mentions(label)
        mention_rec = self._extract_mentions(rec)
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



def eva_one_sentence(encoder, decoder, this_batch):
    this_batch_num = len(this_batch[3])
    this_batch_max_seq = max(this_batch[3])
    last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))

    data = Variable(this_batch[0], volatile=True)
    target = Variable(this_batch[1], volatile=True)
    length = Variable(torch.LongTensor(this_batch[3]), volatile=True)
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size'] / 2),
                   volatile=True)  # encoder gru initial hidden state

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
        if top_path.count(config['X']) == this_batch_max_seq and top_path[-1] == config['EOS_token']:
            break

        if top_path.count(config['X']) > this_batch_max_seq:
            top_path[-1] = config['EOS_token']
            break

    # print beam[0]['paths']  # (output_size, B, 1)
    return top_path


def eva_one_sentence_vib(encoder, decoder, this_batch):
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



# class BatchGetter(object):
#     def __init__(self, file_name):
#         # 游标
#         self.cursor = 0
#
#         train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
#         all_samples = []
#
#         # self.all_samples: [(tokens, labels),()]
#         for line in train_file:
#             line = line.strip()
#             if line:
#                 parts = line.split('|||')
#                 seq = []
#                 for token in parts[0].split(' '):
#                     tokenID = config['WordId'].getID(token.split('#')[0])
#                     seq.append(tokenID)
#                 labels = []
#                 for label in parts[1].split(' '):
#                     labelID = config['OutTags'].getID(label)
#                     labels.append(int(labelID))
#                 all_samples.append((seq, labels))
#         train_file.close()
#         self.all_samples = all_samples
#         self.sample_num = len(self.all_samples)
#         self.reset()
#
#     def __iter__(self):
#         return self
#
#     # 在一个epoch内获得一个batch
#     def next(self):
#         if self.cursor < self.sample_num:
#             required_batch = self.all_samples[self.cursor:self.cursor+1]
#             self.cursor += 1
#             input_seqs = [seq_label[0] for seq_label in required_batch]
#             input_labels = [seq_label[1] for seq_label in required_batch]
#             input_seqs_length = [len(s) for s in input_seqs]
#             input_labels_length = [len(s) for s in input_labels]
#             seqs_padded = [pad_seq(s, max(input_seqs_length)) for s in input_seqs]
#             labels_padded = [pad_seq(s, max(input_labels_length)) for s in input_labels]
#
#             return torch.LongTensor(seqs_padded), torch.LongTensor(labels_padded), input_labels_length, input_seqs_length
#         else:
#             raise StopIteration("out of list")
#
#     # 一个epoch后reset
#     def reset(self):
#         # random.shuffle(self.all_samples)
#         self.cursor = 0

def evaluate_all(my_arg, pr=True):
    emb = LoadEmbedding('res/emb.txt')
    print 'finish loading embedding'
    encoder = CNNEncoder(emb, dropout_p=0)
    decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'],
                                     config['decoder_output_size'], config['decoder_layers'], dropout_p=0)
    en_dict = torch.load('model/encoder_params.pkl')
    de_dict = torch.load('model/decoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    # print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    # decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_getter = BatchGetter('data/eng_dev.txt', 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open('data/eva_result'+str(my_arg), mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_path = eva_one_sentence(encoder, decoder, this_batch)
        top_path = top_path[1:]
        # print [ner_tag.getWord(tag) for tag in top_path]
        evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file, pr)
        if i % 100 == 0:
            print '{} sentences processed'.format(i)
            evaluator.get_performance()

    return evaluator.get_performance()






if __name__ == '__main__':
    emb = LoadEmbedding('res/emb.txt')
    print 'finish loading embedding'
    encoder = CNNEncoder(emb, dropout_p=0)
    decoder = BahdanauAttnDecoderRNN(config, config['encoder_outputs_size'], config['hidden_size'],
                                        config['decoder_output_size'], config['decoder_layers'], dropout_p=0)
    en_dict = torch.load('model0/encoder_params.pkl')
    de_dict = torch.load('model0/decoder_params.pkl')
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    print en_dict.keys()
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    batch_getter = BatchGetter('data/dev', 1, shuffle=False)
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        decoder.cuda(config['cuda_num'])

    ner_tag = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()

    out_file = codecs.open('data/eva_result.txt', mode='wb', encoding='utf-8')

    for i, this_batch in enumerate(batch_getter):
        top_path = eva_one_sentence(encoder, decoder, this_batch)
        top_path = top_path[1:]
        # print [ner_tag.getWord(tag) for tag in top_path]
        evaluator.evaluate(i, this_batch[1].numpy()[0, :].tolist(), top_path, out_file)
        if i % 100 == 0:
            print '{} sentences processed'.format(i)
            evaluator.get_performance()


    evaluator.get_performance()

    # batch_getter = BatchGetter('data/dev')
    # this_batch = next(batch_getter)
    # print this_batch[1].numpy()[0, :].tolist()






