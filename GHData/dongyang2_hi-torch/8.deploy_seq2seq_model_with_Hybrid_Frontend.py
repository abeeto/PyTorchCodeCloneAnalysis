# 第八节
import torch
import torch.nn as nn
import torch.nn.functional as fun

MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):  # 这里的init里没有super，是不是因为这个类是我们自创的不是继承的所以不用写
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {}/{} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                 len(keep_words) / len(self.word2index)))

        # reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.add_word(word)


def normalize_string(s):
    """Lowercase and remove non-letter characters"""
    s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def index_from_sentence(voc, sentence):
    """Takes string sentence, returns sentence of word indexes"""
    # 这里好神奇，pycharm知道我这里的voc是Voc，直接voc. 会出来很多成员变量
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layer=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, dropout=(0 if n_layer == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
        embed = self.embedding(input_seq)  # Convert word indexes to embeddings
        packed = nn.utils.rnn.pack_padded_sequence(embed, input_length)  # Pack padded batch of sequences for RNN module
        output, hidden = self.gru(packed, hidden)  # Forward pass through GRU
        output, _ = nn.utils.rnn.pad_packed_sequence(output)  # Unpack padding
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional GRU outputs
        # Return output and final hidden state
        return output, hidden


class Attn(nn.Module):
    """Attention layer"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate attention method.')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        # 下面这个式子肯定是想逼死我
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_output):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'dot':
            attn_energy = self.dot_score(hidden, encoder_output)
        elif self.method == 'general':
            attn_energy = self.general_score(hidden, encoder_output)
        else:
            attn_energy = self.concat_score(hidden, encoder_output)

        # Transpose max_length and batch_size dimensions
        attn_energy = attn_energy.t()

        # Return the softmax normalized probability scores (with added dimension)
        return fun.softmax(attn_energy, dim=1).unsqueeze(1)


class AttnDecoderRnn(nn.Module):

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layer=1, dropout=0.1):
        super(AttnDecoderRnn, self).__init__()
        self.attn_model = attn_model
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 为什么要在init里面定义这几个层
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, dropout=(0 if n_layer == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_output):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embed = self.embedding(input_step)
        embed = self.embedding_dropout(embed)
        rnn_output, hidden = self.gru(embed, last_hidden)
        attn_weight = self.attn(rnn_output, encoder_output)
        context = attn_weight.bmm(encoder_output.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = fun.softmax(output, dim=1)

        return output, hidden


class GreedySearchDetector(torch.jit.ScriptModule):

    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDetector, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._decoder_n_layers = decoder_n_layers
        self._device = device
        self._SOS_token = SOS_token

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.ScriptModule
    def forward(self, input_seq: torch.Tensor, input_length: torch.Tensor, max_length: int):
        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            # Obtain most likely word token and its softmax score
            decoder_score, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_score), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    index_batch = [index_from_sentence(voc, sentence)]
    lengths = torch.Tensor([len(index) for index in index_batch])
    input_batch = torch.LongTensor(index_batch).transpose(0, 1)
    lengths = lengths.to(device)
    input_batch = input_batch.to(device)

    tokens, scores = searcher(input_batch, lengths, max_length)
    decode_word = [voc.index2word[token.item()] for token in tokens]
    return decode_word


def evaluate_input(searcher, voc):
    while 1:
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            input_sentence = normalize_string(input_sentence)
            output_words = evaluate(searcher, voc, input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot: ', ''.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")


def evaluate_example(sentence, searcher, voc):
    print('> ' + sentence)
    input_sentence = normalize_string(sentence)
    output_words = evaluate(searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot: ', ''.join(output_words))


if __name__ == '__main__':
    import os
    import time
    import re
    import unicodedata
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    print('-' * 15, 'Start', time.ctime(), '-' * 15, '\n')

    device = torch.device("cpu")

    print('%s%s %s %s %s' % ('\n', '-' * 16, 'End', time.ctime(), '-' * 16))
