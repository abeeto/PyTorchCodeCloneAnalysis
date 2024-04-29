from torch import nn
import torch
import numpy as np
import params
from params import args

SOS=params.SOS
EOS=params.EOS
PAD=params.PAD


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        # here input dimention is equal to hidden dimention
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,
                                      hidden_size,
                                      padding_idx=PAD)

        self.gru = nn.GRU(hidden_size, hidden_size)
        # self.gru = nn.RNN(hidden_size, hidden_size)

        self.empty_elem = torch.randn(1, args.stack_elem_size, requires_grad=True)

    def forward(self, inputs, hidden=None, stacks=None):
        # inputs: length * bsz
        # stacks: bsz * nstack * stacksz * stackelemsz
        embs = self.embedding(inputs)
        # inputs(length,bsz)->embd(length,bsz,embdsz)
        hidden=None
        outputs,hidden=self.gru(embs,hidden)

        return outputs, hidden, stacks

    def init_stack(self,batch_size):
        return self.empty_elem.expand(batch_size,
                                      args.nstack,
                                      args.stack_size,
                                      args.stack_elem_size).contiguous()
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size,self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size,
                                      hidden_size,
                                      padding_idx=PAD)
        self.log_softmax=nn.LogSoftmax(dim=1)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # self.gru = nn.RNN(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, stacks=None):
        # input: shape of [bsz]
        # emb: 1 * bsz * embdsz
        emb = self.embedding(input).unsqueeze(0)


        # 1 * bsz * hsz -> bsz * hsz:
        output, hidden = self.gru(emb,hidden)
        output = self.out(output).squeeze(0)
        output = self.log_softmax(output)
        # output: bsz * tar_vacabulary_size

        top1 = output.data.max(1)[1]
        top1 = top1.unsqueeze(1)
        # topv, topi = torch.topk(output,1,dim=1)
        # output_index = topi
        # output_index: bsz * 1

        return output, hidden, top1, stacks
