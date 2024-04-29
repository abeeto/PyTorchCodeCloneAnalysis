from decoder import MultiDecoder
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm
from masked_cross_entropy import masked_cross_entropy
from configurations import config, logger, to_np
from encoder import LoadEmbedding, MultiCNNEncoder
from batch_getter import BatchGetter, get_source_mask
import codecs
import time
torch.manual_seed(1)




def train_iteration(step, encoder, decoder, encoder_optimizer, decoder_optimizer, this_batch):
    # encoder_outputs = Variable(torch.randn(config['max_seq_length'], config['batch_size'], config['hidden_size']))
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    this_batch_num = len(this_batch[2])
    this_batch_max_seq = max(this_batch[2])
    last_hidden = Variable(torch.zeros(config['decoder_layers'], this_batch_num, config['hidden_size']))
    word_input = Variable(torch.zeros(this_batch_num, 1).type(torch.LongTensor))

    data = Variable(this_batch[0])
    target = Variable(this_batch[1])
    length = Variable(torch.LongTensor(this_batch[2]))
    h_0 = Variable(torch.zeros(2, this_batch_num, config['hidden_size']/2))  # encoder gru initial hidden state
    print 'seq_length', max(this_batch[3]), 'label_length', this_batch_max_seq  # (output_size, B, 1)

    if config['USE_CUDA']:
        last_hidden = last_hidden.cuda(config['multi_cuda'][0])
        word_input = word_input.cuda(config['multi_cuda'][0])
        data = data.cuda(config['multi_cuda'][0])
        target = target.cuda(config['multi_cuda'][0])
        length = length.cuda(config['multi_cuda'][0])
        h_0 = h_0.cuda(config['multi_cuda'][0])


    step = Variable(torch.ones(len(config['multi_cuda']), 1)*step)
    h_0 = h_0.transpose(0,1)
    batch_seq_len = Variable(torch.LongTensor(this_batch[3])).unsqueeze(1)
    if config['use_multi']:
        step = step.cuda(config['multi_cuda'][0])
        batch_seq_len = batch_seq_len.cuda(config['multi_cuda'][0])
    encoder_outputs = encoder(step, data, h_0, batch_seq_len)
    encoder_outputs = encoder_outputs.transpose(0, 1)
    # encoder_outputs = encoder_outputs.transpose(1,2)
    # encoder_outputs = encoder_outputs.transpose(0,1)
    source_mask = Variable(get_source_mask(this_batch_num, config['encoder_filter_num'], max(this_batch[3]), this_batch[3]))
    if config['USE_CUDA']:
        source_mask = source_mask.cuda(config['multi_cuda'][0])
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
    encoder_outputs = encoder_outputs.transpose(0, 1)
    seq_label_prob = Variable(torch.zeros(this_batch_max_seq, this_batch_num, config['decoder_output_size']))
    if config['USE_CUDA']:
        seq_label_prob = seq_label_prob.cuda(config['multi_cuda'][0])
    for time_step in range(this_batch_max_seq):
        last_hidden = last_hidden.transpose(0, 1)
        label_prob, cur_hidden, attn_weights = decoder(step, word_input, last_hidden, encoder_outputs)
        cur_hidden = cur_hidden.transpose(0, 1)
        last_hidden = cur_hidden
        seq_label_prob[time_step] = label_prob
        # Choose top word from label_prob
        # value, label = label_prob.topk(1)
        # decoder_out_label.append(label.data)
        # not teacher-forcing
        # word_input = label

        # teacher-forcing
        word_input = target[:, time_step]
    decoder_prob = Variable(torch.FloatTensor([[[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]], [[0,1],[1,0]],[[0,1],[1,0]],[[1,0],[0,1]],[[1,0],[0,1]]]))

    if config['USE_CUDA']:
        decoder_prob = decoder_prob.cuda(config['multi_cuda'][0])

    loss = masked_cross_entropy(seq_label_prob.transpose(0,1).contiguous(), target, length)
    # loss = masked_cross_entropy(F.softmax(decoder_prob.transpose(0,1).contiguous()), target, length)
    print 'loss: ', loss.data[0]
    # logger.scalar_summary('loss', loss.data[0], step.data[0, 0])
    loss.backward()
    clip_grad_norm(decoder.parameters(), config['clip_norm'])
    clip_grad_norm(encoder.parameters(), config['clip_norm'])
    # for tag, value in encoder.named_parameters():
    #     tag = tag.replace('.', '/')
    #     tag = tag.replace('module', 'encoder')
    #     logger.histo_summary(tag, to_np(value), step.data[0, 0])
    #     logger.histo_summary(tag + '/grad', to_np(value.grad), step.data[0, 0])
    # for tag, value in decoder.named_parameters():
    #     tag = tag.replace('.', '/')
    #     tag = tag.replace('module', 'decoder')
    #     logger.histo_summary(tag, to_np(value), step.data[0, 0])
    #     logger.histo_summary(tag + '/grad', to_np(value.grad), step.data[0, 0])
    decoder_optimizer.step()
    encoder_optimizer.step()


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

def train_epoch(epoch, ex_iterations, batch_getter, encoder, decoder, encoder_optimizer, decoder_optimizer):
    batch_getter.reset()
    for iteration, this_batch in enumerate(batch_getter):
        time0 = time.time()
        print 'epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration)
        train_iteration(ex_iterations + iteration, encoder, decoder, encoder_optimizer, decoder_optimizer, this_batch)
        time1 = time.time()
        del this_batch
        print 'this iteration time: ', time1 - time0, '\n'
        # if (ex_iterations + iteration) % config['save_freq'] == 0:
        #     torch.save(decoder.state_dict(), 'model/decoder_params.pkl')
        #     torch.save(encoder.state_dict(), 'model/encoder_params.pkl')
    return ex_iterations + iteration


def train(encoder, decoder, encoder_optimizer, decoder_optimizer):
    batch_getter = BatchGetter('data/train')
    ex_iterations = 6309
    for i in range(1000):
        result = train_epoch(i, ex_iterations, batch_getter, encoder, decoder, encoder_optimizer, decoder_optimizer)
        ex_iterations = result + 1


if __name__ == '__main__':
    print 'ddd'
    emb = LoadEmbedding('res/emb.txt')
    print 'finish loading embedding'
    encoder = MultiCNNEncoder(emb)
    decoder = MultiDecoder(config, config['encoder_outputs_size'], config['hidden_size'],
                                     config['decoder_output_size'], config['decoder_layers'])
    en_dict = torch.load('model/encoder_params.pkl')
    de_dict = torch.load('model/decoder_params.pkl')
    encoder.load_state_dict(en_dict)
    decoder.load_state_dict(de_dict)
    encoder = nn.DataParallel(encoder, device_ids=config['multi_cuda'])
    decoder = nn.DataParallel(decoder, device_ids=config['multi_cuda'])
    # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
    # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
    decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
    encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
    if config['USE_CUDA']:
        encoder.cuda(config['multi_cuda'][0])
        decoder.cuda(config['multi_cuda'][0])
    train(encoder, decoder, encoder_optimizer, decoder_optimizer)
    # torch.save(decoder.state_dict(), 'model/decoder_params.pkl')
    # torch.save(encoder.state_dict(), 'model/encoder_params.pkl')


