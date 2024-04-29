import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class my_word_emb(nn.Module):
    def __init__(self, len_vocab, hidden_dim):
        super(my_word_emb, self).__init__()
        self.word_emb = nn.Embedding(len_vocab, hidden_dim)

    def forward(self, inputs):
        return self.word_emb(inputs)


class my_pos_emb(nn.Module):
    def __init__(self, pos_encoding):
        super(my_pos_emb, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(pos_encoding, freeze=True)

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(0)

        positions.masked_fill_(pos_mask, 0)
        pos_embs = self.pos_emb(positions) # position embedding
        return pos_embs


class ENCODER(nn.Module):
    def __init__(self, args):
        super(ENCODER, self).__init__()
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.layers = nn.ModuleList([my_enc_layer(args) for i in range(args.num_encoder)])

    def forward(self, inputs, attn_mask):
        for enc in self.layers:
            output = enc(srcQ=inputs, srcK=inputs, srcV=inputs, attn_mask = attn_mask)
            inputs = output
        return output

class my_enc_layer(nn.Module):
    def __init__(self, args):
        super(my_enc_layer, self).__init__()
        self.layer = MultiheadAttention(args)
        self.pos_fn = FFNN(args)

    def forward(self, srcQ, srcK, srcV, attn_mask):
        output = self.layer(srcQ, srcK, srcV, attn_mask)
        output = self.pos_fn(output)
        return output


class DECODER(nn.Module):
    def __init__(self, args):
        super(DECODER, self).__init__()
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.layers = nn.ModuleList([my_dec_layer(args) for i in range(args.num_decoder)])

    def forward(self, dec_input, look_ahead_mask, enc_input, attn_mask):
        for dec in self.layers:
            output = dec(dec_input, look_ahead_mask, enc_input, attn_mask)
            dec_input = output
        return output

class my_dec_layer(nn.Module):
    def __init__(self, args):
        super(my_dec_layer, self).__init__()
        self.layer1 = MultiheadAttention(args)
        self.layer2 = MultiheadAttention(args)
        self.pos_fn = FFNN(args)

    def forward(self, dec_input, look_ahead_mask, enc_input, attn_mask):
        output = self.layer1(dec_input, dec_input, dec_input, look_ahead_mask)
        output = self.layer2(output, enc_input, enc_input, attn_mask)
        output = self.pos_fn(output)
        return output

class FFNN(nn.Module):
    def __init__(self, args):
        super(FFNN, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.FFNN_dim = args.FFNN_dim

        self.FFNN = nn.Sequential(
            nn.Linear(self.hidden_dim, self.FFNN_dim),
            nn.ReLU(),
            nn.Linear(self.FFNN_dim, self.hidden_dim)
        )

        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, attention):
        ### FFNN ###
        output = self.FFNN(attention)
        output = self.dropout(output)
        output = output + attention
        output = self.layernorm(output)
        return output

class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()

        # embedding_dim, d_model, 512 in paper
        self.hidden_dim = args.hidden_dim
        # 8 in paper
        self.num_head = args.num_head
        # head_dim, d_key, d_query, d_value, 64 in paper (= 512 / 8)
        self.head_dim = self.hidden_dim // self.num_head
        self.FFNN_dim = args.FFNN_dim
        self.bs = args.batch_size

        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')

        self.fcQ = nn.Linear(self.hidden_dim, self.head_dim * self.num_head)
        self.fcK = nn.Linear(self.hidden_dim, self.head_dim * self.num_head)
        self.fcV = nn.Linear(self.hidden_dim, self.head_dim * self.num_head)
        self.fcOut = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)

        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, srcQ, srcK, srcV, attn_mask):

        ### FIRST ATTENTION ###
        Q = self.fcQ(srcQ) #(self.bs, seq_len, self.num_head * self.head_dim)
        K = self.fcK(srcK) #(self.bs, seq_len, self.num_head * self.head_dim)
        V = self.fcV(srcV) #(self.bs, seq_len, self.num_head * self.head_dim)

        Q = Q.view(self.bs, -1, self.num_head, self.head_dim).transpose(1,2)
        K = K.view(self.bs, -1, self.num_head, self.head_dim).transpose(1,2)
        V = V.view(self.bs, -1, self.num_head, self.head_dim).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)

        scale = 1 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn_prob, V) #(self.bs, self.num_head, -1, self.head_dim)
        context = context.transpose(1, 2).contiguous().view(self.bs, -1, self.num_head * self.head_dim)

        attention = self.fcOut(context) # (self.bs, n_seq, d_hidn)
        attention = self.dropout(attention)
        attention = attention + srcQ
        attention = self.layernorm(attention)

        return attention


class Transformer(nn.Module):
    def __init__(self, conf):
        super(Transformer, self).__init__()

        self.encoder = ENCODER(conf)
        self.decoder = DECODER(conf)

        self.FFNN = nn.Linear(conf.hidden_dim, conf.n_output)
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, conf, enc_inputs, dec_inputs, target):

        word_emb = my_word_emb(len_vocab=5000, hidden_dim=conf.hidden_dim)
        pos_encoding = get_sinusoid_encoding_table(conf.n_seq, conf.hidden_dim)
        pos_encoding = torch.FloatTensor(pos_encoding)
        pos_emb = my_pos_emb(pos_encoding)

        input_sum = make_sum_inputs(conf, word_emb, enc_inputs, pos_emb)
        attn_mask = make_attn_mask(enc_inputs, input_sum)

        dec_input_sum = make_sum_inputs(conf, word_emb, dec_inputs, pos_emb)
        look_ahead_mask = get_attn_decoder_mask(dec_inputs)

        enc_output = self.encoder(enc_inputs, attn_mask)
        dec_output = self.decoder(dec_input_sum, look_ahead_mask, enc_output, attn_mask)

        dec_output = self.FFNN(dec_output)
        dec_output = torch.softmax(dec_output, dim=-1)

        loss = self.criterion(dec_output, target)

        return dec_output
