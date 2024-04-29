# coding: utf-8

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm
from masked_cross_entropy import masked_cross_entropy
import torch.nn.init
from configurations import config, to_np
from my import my_softmax
#
# USE_CUDA = True
# max_seq_length = 4
# max_label_length = 8
# seq_max_len = 4
# batch_size = 2
# hidden_size = 3
# n_layers = 2
# encoder_outputs_dim = 3
# output_size = 2
# cuda_num = 0
# clip_norm = 1
# beam_size = 6

class Attn(nn.Module):

    # 默认encoder 和 decoder hidden_size 相同
    def __init__(self, method, decoder_hidden_size, encoder_outputs_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = decoder_hidden_size
        self.encoder_outputs_size = encoder_outputs_size

        if self.method == 'general':
            linear = nn.Linear(self.encoder_outputs_size, decoder_hidden_size)
            torch.nn.init.uniform(linear.weight, -config['weight_scale'], config['weight_scale'])
            torch.nn.init.constant(linear.bias, 0)
            self.attn = linear

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, decoder_hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, decoder_hidden_size))

    # hidden: (1, B, hidden) encoder_outputs: (S, B, encoder_hidden)
    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        hidden = hidden.transpose(0, 1).contiguous()  # (B, 1, hidden)
        encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()  # (B, S, encoder_hidden)
        encoder_pro = self.attn(encoder_outputs)  # (B, S, hidden)
        attn_energies = torch.bmm(hidden, encoder_pro.transpose(1, 2))  # (B, 1, S)
        attn_energies = my_softmax(attn_energies, 2)
        return attn_energies




        # # Create variable to store attention energies
        # attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S
        #
        # if config['USE_CUDA']:
        #     attn_energies = attn_energies.cuda(hidden.get_device())
        #
        # # For each batch of encoder outputs
        # for b in range(this_batch_size):
        #     # Calculate energy for each encoder output
        #     for i in range(max_len):
        #         attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))
        #
        # # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        # return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':

            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class BahdanauAttnDecoderRNN(nn.Module):

    def __init__(self, config, encoder_outputs_dim, hidden_size, output_size, n_layers=1, dropout_p=config['dropout']):
        super(BahdanauAttnDecoderRNN, self).__init__()

        self.encoder_outputs_dim = encoder_outputs_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.decoder_embedding = nn.Embedding(output_size, hidden_size)
        torch.nn.init.uniform(self.decoder_embedding.weight, -config['weight_scale'], config['weight_scale'])
        self.dropout = nn.Dropout(dropout_p)
        self.att = Attn(config['att_mode'], hidden_size, encoder_outputs_dim)
        decoder_gru = nn.GRU(input_size=encoder_outputs_dim + hidden_size, hidden_size=hidden_size,
                             num_layers=n_layers, dropout=dropout_p)
        for name, param in decoder_gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)
        self.decoder_gru = decoder_gru
        # out_linear = nn.Linear(encoder_outputs_dim + hidden_size, output_size)
        out_linear = nn.Linear(hidden_size, output_size)
        torch.nn.init.uniform(out_linear.weight, -config['weight_scale'], config['weight_scale'])
        torch.nn.init.constant(out_linear.bias, 0)
        self.out = out_linear

    # word_input: (batch, 1)  last_hidden: (layers * directions, batch, hidden) encoder_outputs: (S, B, hidden)
    # one time-step
    def forward(self, step, word_input, last_hidden, encoder_outputs):
        self.decoder_gru.flatten_parameters()
        embedded = self.decoder_embedding(word_input)
        embedded = embedded.view(1, -1, self.hidden_size)
        embedded = self.dropout(embedded) # size(1, B, hidden)

        attn_weights = self.att(last_hidden[-1].unsqueeze(0), encoder_outputs)  # B x 1 x S
        # logger.histo_summary('att_weights', to_np(attn_weights), step)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        context = context.transpose(0, 1)  # 1 x B x encoder_hidden

        rnn_input = torch.cat((embedded, context), 2) # (1, B, hidden + encoder_hidden)
        # output: (1, B, hidden)  cur_hidden: (num_layers * num_directions, B, hidden)
        output, cur_hidden = self.decoder_gru(rnn_input, last_hidden)

        # cat = torch.cat((output.squeeze(0), context.squeeze(0)), 1) # (B, hidden + encoder_hidden)
        # label_prob = self.out(cat)  # (B, output_size)
        label_prob = self.out(output.squeeze(0))

        # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)
        return label_prob, cur_hidden, attn_weights

class MultiDecoder(BahdanauAttnDecoderRNN):
    def __init__(self, config, encoder_outputs_dim, hidden_size, output_size, n_layers=1, dropout_p=config['dropout']):
        super(MultiDecoder, self).__init__(config, encoder_outputs_dim, hidden_size, output_size,
                                           n_layers=n_layers, dropout_p=config['dropout'])

    # word_input: (batch, 1)  last_hidden: (batch, layers * directions, hidden) encoder_outputs: (B, S, hidden)
    # one time-step
    def forward(self, step, word_input, last_hidden, encoder_outputs):
        step = step.data[0, 0]
        last_hidden = last_hidden.transpose(0, 1).contiguous()
        encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()
        label_prob, cur_hidden, attn_weights = super(MultiDecoder, self).forward(step, word_input, last_hidden, encoder_outputs)
        return label_prob, cur_hidden.transpose(0, 1), attn_weights



if __name__ == '__main__':
    pass

    # encoder_outputs = Variable(torch.randn(max_seq_length, batch_size, hidden_size))
    # last_hidden = Variable(torch.randn(n_layers, batch_size, hidden_size))
    # decoder = BahdanauAttnDecoderRNN(encoder_outputs_dim, hidden_size, output_size, n_layers)
    # optimizer = torch.optim.Adadelta(decoder.parameters())
    # word_input = Variable(torch.LongTensor([[0], [1]]))
    # if USE_CUDA:
    #     encoder_outputs = encoder_outputs.cuda(cuda_num)
    #     last_hidden = last_hidden.cuda(cuda_num)
    #     word_input = word_input.cuda(cuda_num)
    #     decoder.cuda(cuda_num)


    # train
    # decoder_out_label = []
    # seq_label_prob = Variable(torch.zeros(max_label_length, batch_size, output_size))
    # if USE_CUDA:
    #     seq_label_prob = seq_label_prob.cuda()
    # for time_step in range(max_label_length):
    #     label_prob, cur_hidden, attn_weights = decoder(word_input, last_hidden, encoder_outputs)
    #     last_hidden = cur_hidden
    #     seq_label_prob[time_step] = label_prob
    #     # Choose top word from label_prob
    #     value, label = label_prob.topk(1)
    #     decoder_out_label.append(label.data)
    #     word_input = label
    #
    # # print decoder_out_label # (output_size, B, 1)
    # decoder_prob = Variable(torch.FloatTensor([[[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]], [[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]]]))
    # target = Variable(torch.LongTensor([[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]]))  # (batch, max_label_length)
    # length = Variable(torch.LongTensor([5,8]))
    # if USE_CUDA:
    #     decoder_prob = decoder_prob.cuda(cuda_num)
    #     target = target.cuda(cuda_num)
    #     length = length.cuda(cuda_num)
    # loss = masked_cross_entropy(F.softmax(seq_label_prob.transpose(0,1).contiguous()), target, length)
    # # loss = masked_cross_entropy(F.softmax(decoder_prob.transpose(0,1).contiguous()), target, length)
    # print loss
    # optimizer.zero_grad()
    # loss.backward()
    # clip = clip_grad_norm(decoder.parameters(), clip_norm)
    # optimizer.step()


    # evaluate
    # beam = [{'paths':[[],[]], 'prob':Variable(torch.zeros(batch_size, 1)),
    #          'hidden':Variable(torch.randn(n_layers, batch_size, hidden_size))},{}]  # beam_size*batch_size*([path],hidden)
    # beam = []
    # for beam_i in range(beam_size):
    #     prob_init = Variable(torch.zeros(batch_size, 1))
    #     hidden_init = Variable(torch.randn(n_layers, batch_size, hidden_size))
    #     if USE_CUDA:
    #         prob_init = prob_init.cuda(cuda_num)
    #         hidden_init = hidden_init.cuda(cuda_num)
    #     one_beam = {'paths':[], 'prob': prob_init, 'hidden': hidden_init}
    #     for batch_i in range(batch_size):
    #         one_beam['paths'].append([0])
    #     beam.append(one_beam)



    # beam = [{'paths':[] , 'tails':range(output_size) },{'paths':[] , 'tails':range(output_size)}]

    # print beam
    # for time_step in range(max_label_length):
    #     # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)\
    #
    #     next_prob = []
    #     cur_hidden_lst = []
    #     for i, beam_i in enumerate(beam):
    #         word_input = Variable(torch.LongTensor(batch_size, 1).zero_())
    #         for batch_i in range(len(beam_i['paths'])):
    #             word_input[batch_i, 0] = beam_i['paths'][batch_i][-1]
    #         last_hidden = beam_i['hidden']
    #         if USE_CUDA:
    #             word_input = word_input.cuda(cuda_num)
    #             last_hidden = last_hidden.cuda(cuda_num)
    #         # word_input: (batch, 1)  last_hidden: (layers * directions, batch, hidden) encoder_outputs: (S, B, hidden)
    #         # label_prob: (B, output_size)  cur_hidden: (num_layers * num_directions, B, hidden)  att_weights: (B, 1, S)
    #         label_prob, cur_hidden, attn_weights = decoder(word_input, last_hidden, encoder_outputs)
    #         cur_hidden_lst.append(cur_hidden)
    #         log_label_prob = F.log_softmax(label_prob)
    #         next_prob.append(beam_i['prob'].expand_as(log_label_prob) + log_label_prob) # (batch_size, output_size)
    #     cat = torch.cat(next_prob, 1)  # (batch, outputs_size*beam_size)
    #     # indices:(batch, beam_size)
    #     value, indices = cat.topk(beam_size, 1)
    #     beam_num = indices / int(output_size)  # (batch, beam_size)
    #     label_num = indices - beam_num * output_size  # (batch, beam_size)
    #     beam_num = beam_num.cpu().data
    #     label_num = label_num.cpu().data
    #     new_beam = []
    #     for beam_i in range(beam_size):
    #         prob_init = Variable(torch.zeros(batch_size, 1))
    #         hidden_init = Variable(torch.randn(n_layers, batch_size, hidden_size))
    #         if USE_CUDA:
    #             prob_init = prob_init.cuda(cuda_num)
    #             hidden_init = hidden_init.cuda(cuda_num)
    #         one_beam = {'paths': [], 'prob': prob_init, 'hidden': hidden_init}
    #         new_beam.append(one_beam)
    #     for i_batch in range(batch_size):
    #         for i_beam in range(beam_size):
    #             this_beam_num = beam_num[i_batch][i_beam]
    #             this_label = label_num[i_batch][i_beam]
    #             a = beam[this_beam_num]['paths'][i_batch][:]
    #             a.append(this_label)
    #             new_beam[i_beam]['paths'].append(a)
    #             new_beam[i_beam]['hidden'] = cur_hidden_lst[this_beam_num]
    #             new_beam[i_beam]['prob'][i_batch, 0] = next_prob[this_beam_num][i_batch, this_label]
    #     beam = new_beam

    # print beam[0]['paths']  # (output_size, B, 1)
    # decoder_prob = Variable(torch.FloatTensor(
    #     [[[0, 1], [1, 0]], [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, 1], [1, 0]],
    #      [[1, 0], [0, 1]], [[1, 0], [0, 1]]]))
    # target = Variable(
    #     torch.LongTensor([[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]]))  # (batch, max_label_length)
    # length = Variable(torch.LongTensor([5, 8]))
    # if USE_CUDA:
    #     target = target.cuda()
    #     length = length.cuda()
    # loss = masked_cross_entropy(seq_label_prob.transpose(0, 1).contiguous(), target, length)
    # optimizer.zero_grad()
    # loss.backward()
    # clip = clip_grad_norm(decoder.parameters(), clip_norm)
    # optimizer.step()
    # a = Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))
    # print a.topk(2, 1)


















    # a = Variable(torch.FloatTensor([[0.9,0.1],[0.8,0.2],[0.3,0.7]]))
    # print a.topk(1)






    # label_prob, cur_hidden, attn_weights = decoder(word_input, last_hidden, encoder_outputs)
    # optimizer.zero_grad()
    # s = cur_hidden.sum()
    # print decoder.parameters().next().grad, s
    # clip = clip_grad_norm(decoder.parameters(), clip_norm)
    # s.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # label_prob, cur_hidden, attn_weights = decoder(word_input, last_hidden, encoder_outputs)
    # s = cur_hidden.sum()
    # print label_prob, decoder.parameters().next().grad, s






