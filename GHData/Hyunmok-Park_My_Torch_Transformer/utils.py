import torch
#import sentencepiece
from pos_embedding import *
from net import *

def make_input_tensor(input):
    # vocab loading
    vocab_file = "<path of data>/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    # 입력 texts
    lines = [
        "겨울은 추워요.",
        "감기 조심하세요."
    ]

    # text를 tensor로 변환
    inputs = []
    for line in lines:
        pieces = vocab.encode_as_pieces(line)
        ids = vocab.encode_as_ids(line)
        inputs.append(torch.tensor(ids))
        print(pieces)

    # 입력 길이가 다르므로 입력 최대 길이에 맟춰 padding(0)을 추가 해 줌
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs

def make_sum_inputs(conf, word_emb, inputs, pos_emb):
    inputs_embs = word_emb(inputs)
    pos_embs = pos_emb(inputs)
    input_sum = inputs_embs + pos_embs
    return input_sum

def make_attn_mask(inputs):
    attn_mask = inputs.eq(0).unsqueeze(1).expand(input_sum.size(0), input_sum.size(1), input_sum.size(1))
    return attn_mask

def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask

