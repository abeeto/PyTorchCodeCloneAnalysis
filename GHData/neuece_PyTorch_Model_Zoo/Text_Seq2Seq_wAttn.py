# Seq2Seq with attention

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, opts, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(input_size,embed_size)  
        
        if opts.word_Embedding:
            self.embedding.weight.data.copy_(opts.pretrained_weight)
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.dropout = dropout        
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # input_seqs: Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        # input_lenghts: list of sequence length
        # hidden: initial state of GRU

        embedded = self.embedding(input_seqs)
        # pack to the same length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # feed data to gru/lstm
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  
        return outputs, hidden

class DynamicEncoder(nn.Module):
    
    def __init__(self, input_size, embed_size, hidden_size, opts, n_layers=1, dropout=0.5):
        super().__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(input_size,embed_size)  
        
        if opts.word_Embedding:
            self.embedding.weight.data.copy_(opts.pretrained_weight)
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.dropout = dropout        
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lens, hidden=None):

        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        # sorting
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        # packing
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        # feed to gru/lstm
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        # hidden: previous hidden state of the decoder, in shape (layers*directions,B,H)
        # encoder_outputs: encoder outputs from Encoder, in shape (T,B,H)
        # param src_len: used for masking. NoneType or tensor in shape (B) indicating sequence length

        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        
        # compute attention score
        attn_energies = self.score(H,encoder_outputs) 
        
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)

        output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden
    

class LuongAttnDecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0):

        super(LuongAttnDecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embed_size)
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        # hc: [hidden, context]
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        # s: softmax
        self.Ws = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        gru_out, hidden = self.gru(embedded, hidden)

        attn_prod = torch.mm(self.attn(hidden)[0], encoder_outputs.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_outputs)

        # hc: [hidden: context]
        hc = torch.cat([hidden[0], context], dim=1)
        out_hc = F.tanh(self.Whc(hc))
        output = F.log_softmax(self.Ws(out_hc), dim=1)

        return output, hidden, attn_weights