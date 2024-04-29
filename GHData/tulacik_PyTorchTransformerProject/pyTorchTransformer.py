# Resources:
# https://www.programmersought.com/article/499733903/
# https://www.programmersought.com/article/51571204815/
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# https://www.youtube.com/watch?v=M6adRGJe5cQ
# https://www.youtube.com/watch?v=9sHcLvVXsns
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/seq2seq_transformer
# https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# https://github.com/SamLynnEvans/Transformer
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# http://peterbloem.nl/blog/transformers
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/




import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import urllib.request
import math
import random
import pandas as pd
import io
from torch import nn
from torchtext.data import Field, Iterator
from torchtext.datasets import TranslationDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
# from einops import rearrange, reduce, repeat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text = ''
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# encode the text and map each character to an integer and vice versa
#chars = tuple(set(text))
#int2char = dict(enumerate(chars))
#char2int = {ch: ii for ii, ch in int2char.items()}

# test encode the text
#encoded = np.array([char2int[ch] for ch in text])
#print(encoded[:100])

# tokenizer as per lecture notes
def Transformer_dataloader(datasource_folder_path, batch_size=32, batch_first=True):
    split_chars = lambda x: list(x) # keeps whitespaces
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_field_x = Field(tokenize=split_chars,
        init_token='<sos>',
        eos_token='<eos>',
        pad_token='<pad>',
        lower=False,
        batch_first=batch_first)
    # train_field_y = Field(tokenize=split_chars,
    #     init_token='<sos>',
    #     eos_token='<eos>',
    #     pad_token='<pad>',
    #     lower=False,
    #     batch_first=batch_first)

    # train_dataset_x = TranslationDataset(
    #     path=train_path_x,
    #     text_field=train_field)
    # validate_dataset_x = TranslationDataset(
    #     path=validation_path_x,
    #     text_field=train_field)
    # train_dataset_y = TranslationDataset(
    #     path=train_path_y,
    #     text_field=train_field)
    # validate_dataset_y = TranslationDataset(
    #     path=validation_path_y,
    #     text_field=train_field)
    TRAIN_FILE_NAME = "train"
    VALID_FILE_NAME = "interpolate"
    INPUTS_FILE_ENDING = ".x"
    TARGETS_FILE_ENDING = ".y"
    # print("passed in folder path is {}".format(datasource_folder_path))
    train_dataset, valid_dataset = TranslationDataset.splits(
        path='./', 
        # datasource_folder_path,
        root= './', # datasource_folder_path,
        exts=(INPUTS_FILE_ENDING, TARGETS_FILE_ENDING),
        fields=(train_field_x, train_field_x),
        train=TRAIN_FILE_NAME,
        validation=VALID_FILE_NAME,
        test=None)
    


    # Using the generated dataset object ,
    # create the iterator...

    train_iterator = Iterator(
        dataset=train_dataset,
        batch_size=batch_size,  # TODO: batch size
        train=True,
        repeat=False,
        shuffle=True,
        device=device)  # TODO: device

    validate_iterator = Iterator(
        dataset=valid_dataset,
        batch_size=batch_size,
        train=True,
        repeat=False,
        shuffle=True,
        device=device)
    
    # build vocab, constructs train_field.vocab
    train_field_x.build_vocab(train_dataset.src)
    # train_field_y.build_vocab(train_dataset.trg)
    print("train src vocab size: {}".format(len(train_field_x.vocab)))
    print("All chars in src vocabulary:{}".format("".join([train_field_x.vocab.itos[i] for i in range(len(train_field_x.vocab))])))
    # print("train trg vocab size: {}".format(len(train_field_y.vocab)))
    # print("All chars in trg vocabulary:{}".format("".join([train_field_y.vocab.itos[i] for i in range(len(train_field_y.vocab))])))
    # train_iter, val_iter, test_iter = data.BPTTIterator.splits(
    #    (train, val, test), batch_size=batch_size, device=-1, bptt_len=32, repeat=False, shuffle=False)

    # print(len(validate_field.vocab))
    # train_iter, val_iter, test_iter = data.Iterator.splits(
    #     (train, val, test), sort_key=lambda x: len(x.Text),
    #     batch_sizes=(32, 256, 256), device=-1)
    # return train_field_x, train_field_y, train_iterator, validate_iterator
    return train_field_x, train_iterator, validate_iterator


def file_statistics(data_files):
    for abs_path in data_files:
        with open(abs_path, "r") as f:
            lines = f.readlines()
            print(abs_path + " sentences: " + str(len(lines)))
            print(abs_path + " avg length: " + str(sum(len(line) for line in lines)/len(lines)))
            max_len  = 0
            for line in lines:
                if len(line) >= max_len:
                    max_len = len(line)
                
            print(f"max lenght of a sentence in {abs_path} is: {max_len}")



def get_files_paths():
    cur_dir = os.path.dirname(__file__)
    file_path_validate_q = "interpolate.x"
    file_path_validate_a = "interpolate.y"
    file_path_validate_qa = "interpolate.xy"
    file_path_train_q = "train.x"
    file_path_train_a = "train.y"
    file_path_train_qa = "train.xy"
    abs_path_v_q = os.path.join(cur_dir, file_path_validate_q)
    abs_path_v_a = os.path.join(cur_dir, file_path_validate_a)
    abs_path_v_qa = os.path.join(cur_dir, file_path_validate_qa)
    abs_path_t_q = os.path.join(cur_dir, file_path_train_q)
    abs_path_t_a = os.path.join(cur_dir, file_path_train_a)
    abs_path_t_qa = os.path.join(cur_dir, file_path_train_qa)
    data_files = []
    data_files.append(abs_path_v_q)
    data_files.append(abs_path_v_a)
    data_files.append(abs_path_v_qa)
    data_files.append(abs_path_t_q)
    data_files.append(abs_path_t_a)
    data_files.append(abs_path_t_qa)
    return data_files

def save_model_state(state, filename="current_best_model_state.pth.tar"):
    print("Saving current state")
    torch.save(state, filename)


def load_model_state(model_state, model, optimizer):
    print("Loading selected model state")
    model.load_state_dict(model_state["state_dict"])
    optimizer.load_state_dict(model_state["optimizer"])





class PositionalEncoding(nn.Module):   
    # function to positionally encode src and target sequencies 
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyTransformerModel(nn.Module):
    # should implement init and forward function
    # define separate functions for masks
    # define forward function with
    # implement:
    #  embedding layer
    #  positional encoding
    #  encoder layer
    #  decoder layer
    #  final classification layer
    # encoder -> forward once
    # decoder -> forward multiple times (for one encoder forward)
    # each output of decoder must be concatenated to input dec_input = torch.cat([dec_input], [dec_output])
    # early stopping => <eos> ?? attention you are working with batches!
    # 3 encoder layers, 2 decoder layers, hidden dimension of 256, feed-forward-dimension of 1024, using 8 heads
    def __init__(self, d_model = 512, vocab_length = 30, sequence_length = 512, num_encoder_layers = 3, num_decoder_layers = 2, num_hidden_dimension = 256, feed_forward_dimensions = 1024, attention_heads = 8, dropout = 0.1, pad_idx = 1, eos_idx = 3, device = "CPU", batch_size = 32):
        #, ninp, device, nhead=8, nhid=2048, nlayers=2, dropout=0.1, src_pad_idx = 1, max_len=5000, forward_expansion= 4):
        super(MyTransformerModel, self).__init__()
        # embedding > create embedding vector for each entry in vocabulary -> vocab length
        # holds extra information about each entry in vocabulary -> gives context to entries
        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # d_model == number of inputs ?? -> sequence length (max sequence length)
        self.src_embedding = nn.Embedding(vocab_length, d_model)
        # wrong implementation before
        # self.src_embedding = nn.Embedding(vocab_length, embedding_length)
        # positinal encoding -> necessary to avoid positional invariance of transformer
        # must be of same length as embedding.
        # embedding and positional encoding will be "summed" -> therefor size is sequence length x embedding_length
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # encoder self-attention -> sees all in src
        # decoder self-attention -> sees itself and previous 
        # decoder to encoder -> attend to all src positions
        # use in forward fn masks
        # create masks (6 types)
        # padding masks => 
        # src_key_padding_mask => mask padding of source
        # trg_key_padding_mask => mask padding of target
        # memory_key_padding_mask => encoder to decoder padding (equal to src_key_padding_mask)
        # attention masks =>
        # src_mask => not necessary in this case => define as None
        # trg_mask => use generate_square_subsequent_mask
        # memory_mask => non necessary in this case => define as None
        # attention mask -> mask (-inf before softmax) all position that should be "ignored" e.g. padding or not yet attended (time step)
        #
        #
        #
        self.vocab_length = vocab_length
        self.d_model = d_model
        self.src_mask = None # attention mask
        self.memory_mask = None # attention mask
        #
        # num_encoder_layers = 3, num_decoder_layers = 2, num_hidden_dimension = 256, feed_forward_dimensions = 1024, attention_heads = 8
        # is this "separate" declaration necessary?
        # encoder_layers = TransformerEncoderLayer(ninp, attention_heads, num_hidden_dimension, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # decoder_layers = TransformerDecoderLayer()
        # self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        # self.decoder = nn.Linear(ninp, ntoken)
        self.pad_idx = pad_idx        
        self.eos_idx = eos_idx        
        self.device = device        
        # def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, 
        # num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
        # activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None)
        self.batch_size = batch_size
        self.transformer = nn.Transformer(
            d_model,
            attention_heads,
            num_encoder_layers,
            num_decoder_layers,
            feed_forward_dimensions,
            dropout,
        )

        self.fc = nn.Linear(d_model, vocab_length)
        # self.init_weights() <= used in tutorial

    # already defined in the function generate_square_subsequent_mask
    # def _generate_square_subsequent_mask(self, sz):
    #    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #    return mask

    def src_att_mask(self, src_len):
        mask = (torch.triu(torch.ones(src_len, src_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def no_peak_att_mask(self, batch_size, src_len, time_step):
        mask = np.zeros((batch_size, src_len), dtype=bool)
        mask[:, time_step: ] = 1 # np.NINF
        mask = torch.from_numpy(mask)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_mem_mask(self, trg_len, src_len):
        mask = (torch.triu(torch.ones(src_len, trg_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_src_key_padding_mask(self, src):
        # mask "<pad>"
        src_mask = src.transpose(0, 1) == self.pad_idx ## OLD CODE -> transpose should be wrong?! expected src_key_padding_mask: (N, S) <= batch, src
        # src_mask = src == self.pad_idx
        # (N, src_len)
        return src_mask.to(self.device)

    def make_trg_key_padding_mask(self, trg):
        # same as above -> expected tgt_key_padding_mask: (N, T)
        tgt_mask = trg.transpose(0, 1) == self.pad_idx
        # tgt_mask = trg == self.pad_idx
        # (N, src_len)
        return tgt_mask.to(self.device)


    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    
    
    def forward(self, src, trg):
        # def forward(self, src, trg, criterion, losses, total_loss, writer, step):
        # should 
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape
        output_fc = trg        
        # src_positions = (
        #     torch.arange(0, src_seq_length)
        #     .unsqueeze(1)
        #     .expand(src_seq_length, N)
        #     .to(self.device)
        # )
        # trg_positions = (
        #     torch.arange(0, trg_seq_length)
        #     .unsqueeze(1)
        #     .expand(trg_seq_length, N)
        #     .to(self.device)
        # )
        #  S - source sequence length
        #  T - target sequence length
        #  N - batch size
        #  E - feature number
        #  src: (S, N, E) (sourceLen, batch, features)
        #  tgt: (T, N, E)
        #  src_mask: (S, S)
        #  tgt_mask: (T, T)
        #  memory_mask: (T, S)
        #  src_key_padding_mask: (N, S)
        #  tgt_key_padding_mask: (N, T)
        #  memory_key_padding_mask: (N, S)
        # encoder part
        # should create necessary masks and input for encoder
        # src = rearrange(src, 'n s -> s n')
        #tensor_src = NamedTensor(src, ("n", "s"))
        src = src.permute(1, 0).to(self.device)
                # print("src shape {}".format(src.shape))        
        # print(src)
        # src = self.pos_enc(src * math.sqrt(self.d_model)) 
        embed_src = self.src_embedding(src)        
        # print("embed_src shape {}".format(embed_src.shape))
        # print(embed_src)
        position_embed_src =  self.pos_encoder(embed_src)
        # position_embed_src = position_embed_src.transpose(0, 1)
        #print("position_embed_src shape {}".format(position_embed_src.shape))
        #print(position_embed_src)


        # get encoder output
        # forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None)
        # forward encoder just once for a batch
        # attention forward of encoder expects => src, src_mask, src_key_padding_mask +++ possible positional encoding error !!!
        src_padding_mask = self.make_src_key_padding_mask(src)
        # print("KEY - src_padding_mask shape {}".format(src_padding_mask.shape))
        # print("should be of shape: src_key_padding_mask: (N, S)")
        # print(src_padding_mask)
        encoder_output = self.transformer.encoder.forward(position_embed_src, src_key_padding_mask = src_padding_mask)
        # print("encoder_output")  
        # print("encoder_output shape {}".format(encoder_output.shape))
        # print(encoder_output)  

        # end of encoder
        mem_padding_mask = self.make_src_key_padding_mask(src)       
        loss_fc_return = 0
        
        time_step = 1
        N, max_trg_seq_length = trg.shape
        trg_start = trg[:, :1]
        trg_in = trg[:, :1]
        current_length = 1

        # start of decoder
        # trg_in = rearrange(trg_in, 'n t -> t n')
        # trg_in_tensor = NamedTensor(trg_in, ("n", "t"))
        trg_in = trg_in.permute(1, 0).to(self.device)
        # trg_start = rearrange(trg_start, 'n t -> t n')
        trg_start = trg_start.permute(1, 0).to(self.device)        
        decoder_outputs = torch.zeros(self.batch_size, max_trg_seq_length, self.vocab_length).to(self.device)
        while current_length < max_trg_seq_length :
            # trg_out = trg[:, 1:]
            # print("trg_in shape {}".format(trg_in.shape))
            # print(trg_in)
            embed_trg = self.src_embedding(trg_in).to(self.device)
            # print("embed_trg shape {}".format(embed_trg.shape))
            # print(embed_trg)
            position_embed_trg = self.pos_encoder(embed_trg).to(self.device)
            # position_embed_trg = position_embed_trg.transpose(0, 1)
            # print("position_embed_trg shape {}".format(position_embed_trg.shape))
            # print(position_embed_trg)
            trg_padding_mask = self.make_trg_key_padding_mask(trg_in).to(self.device)
            # print("KEY - trg_padding_mask shape {}".format(trg_padding_mask.shape))
            # print("should be of shape: trg_key_padding_mask: (N, T)")
            # print(trg_padding_mask)
            trg_mask = self.transformer.generate_square_subsequent_mask(current_length).to(self.device)
            # print("trg_mask shape {}".format(trg_mask.shape))
            # print("trg_mask should be of shape tgt_mask: (T, T)")
            # print(trg_mask)
            # att_mask = self.src_att_mask(trg_seq_length).to(self.device)
            # error => memory_mask: expected shape! (T, S) !!! this is not a key_padding_mask!
            # att_mask = self.no_peak_att_mask(self.batch_size, src_seq_length, time_step).to(self.device)
            # print("att_mask shape {}".format(att_mask.shape))
            # print("att_mask should be of shape  memory_mask: (T, S)")
            # print(att_mask)
            att_mask = self.generate_mem_mask(current_length, src_seq_length).to(self.device)
            # print("att_mask shape {}".format(att_mask.shape))
            # print("att_mask should be of shape  memory_mask: (T, S)")
            # print(att_mask)

            # forward decoder till all in batch did not reach <eos>?
            # def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
            # memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
            # memory_key_padding_mask: Optional[Tensor] = None)
            # first forward        
            decoder_output = self.transformer.decoder.forward(position_embed_trg, encoder_output, trg_mask, att_mask, trg_padding_mask, mem_padding_mask)
            # what next?! Should repeat till 
            # should generate output till all predictions have reached eos or max_length is reached
            # 
            # print("decoder_output")  
            # print("decoder_output shape {}".format(decoder_output.shape))
            # print(decoder_output)
            # output = rearrange(decoder_output, 't n e -> n t e')
            output = decoder_output.permute(1, 0, 2).to(self.device)
            output_fc = self.fc(output)
            

            # decoder_outputs[current_length -1] = output_fc
            # print("output")  
            # print("output shape {}".format(output.shape))
            # print(output)


            # predicted = F.softmax(output, dim=-1)
            # print("predicted")  
            # print("predicted shape {}".format(predicted.shape))
            # print(predicted)
            # top k
            top_value, top_index = torch.topk(output_fc, k=1, sorted=False)
            # top_index = torch.squeeze(top_index)
            # print("top_index before rearrange - shape {}".format(top_index.shape))
            # print(top_index)
            # top_index = rearrange(top_index, 'n t e-> t (n e)')
            top_index = top_index.permute(1, 0, 2).to(self.device)
            top_index = torch.flatten(top_index, start_dim= 1).to(self.device)
            # print("top_index shape {}".format(top_index.shape))
            # print(top_index)
            # output_aio = self.transformer(position_embed_src, position_embed_trg, tgt_mask=trg_mask, src_key_padding_mask=src_padding_mask,
            #                           tgt_key_padding_mask=trg_padding_mask, memory_key_padding_mask=mem_padding_mask)
            # output_fc_aio =  self.fc(output_aio)
            # output_fc_aio = F.softmax(output_fc_aio, dim=-1)
            # print("aio")
            # print("aio shape {}".format(output_fc_aio.shape))
            # print(output_fc_aio)
            # top_value_aio, top_index_aio = torch.topk(output_fc_aio, k=1, sorted=False)
            # top_index_aio = torch.squeeze(top_index_aio)
            # print("top_index shape {}".format(top_index_aio.shape))
            # print(top_index_aio)
            # print("trg")  
            # trg_rear = rearrange(trg, 't n -> n t')
            # print(trg)
            trg_in = torch.cat((trg_start, top_index), 0)            
            current_length = current_length + 1
            #               target_i = trg[:, 1:current_length]
            #               # output_fc = rearrange(output_fc, 'b o e-> (b o) e')
            #               #print("shape output_fc {} ".format(output_fc.shape))
            #               output_fc_flatten = torch.flatten(output_fc, start_dim=0, end_dim=1).to(device)
            #               #print("shape output_fc_flatten {} ".format(output_fc_flatten.shape))
            #               target_i_flatten = torch.flatten(target_i).to(device)
            #               #print("shape target_i {} ".format(target_i.shape))
            #               #print("shape target_i_flatten {} ".format(target_i_flatten.shape))
            #               output_fc_flatten = output_fc_flatten.float()
            #               loss = criterion(output_fc_flatten, target_i_flatten)
            #               losses.append(loss.item())
            #               total_loss += loss.item()
            #               # Back prop
            #               loss.backward(retain_graph=True)
            #               writer.add_scalar("Training loss", loss, global_step=step)            
            # should check for early stopping -> each sentence in trg_in contains <eos> token 
        
        # print("top_value")  
        # print("top_value shape {}".format(top_value.shape))
        # print(top_value)
        #trg_in = rearrange(trg_in, 't n -> n t')
        trg_in = trg_in.transpose(0, 1).to(self.device)
        return trg_in, output_fc, decoder_outputs # , criterion, losses, total_loss, writer, step




def create_train_transformer(train_field_x, train_iterator, validation_iterator, batch_size, num_epochs):
    save_model = False
    # Training hyperparameters
    num_epochs = num_epochs
    learning_rate = 1e-4
    batch_size = batch_size
    vocab_train_x = train_field_x
    print_every = 600
    optimizer_step = 10
    last_validation_accuracy = 0
    # Model hyperparameters
    src_vocab_size = len(train_field_x.vocab)
    # trg_vocab_size = len(train_field_y.vocab)
    d_model = 512
    max_sequence_length = 50
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 2
    dropout = 0.10
    num_hidden_dimension = 256
    forward_expansion = 1028
    src_pad_idx = train_field_x.vocab.stoi["<pad>"]
    src_eos_idx = train_field_x.vocab.stoi["<eos>"]
    # Tensorboard to plot loss
    writer = SummaryWriter("runs/loss_plot")
    step = 0
    # init: vocab_length = 30, sequence_length = 512, num_encoder_layers = 3, num_decoder_layers = 2, 
    # num_hidden_dimension = 256, feed_forward_dimensions = 1024, attention_heads = 8, dropout = 0.1, pad_idx = 3, device = "CPU"
    model = MyTransformerModel(
        d_model,
        src_vocab_size,
        max_sequence_length,
        num_encoder_layers,
        num_decoder_layers,
        num_hidden_dimension,
        forward_expansion,
        num_heads,
        dropout,
        src_pad_idx,
        src_eos_idx,
        device,
        batch_size
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    # pad_idx = train_field_x.vocab.stoi["<pad>"]
    # criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx).to(device)
    criterion = nn.CrossEntropyLoss().to(device) # is this necessart to ensure that there is only pad on the end
    # critetion = nn.MSELoss().to(device)
    losses = []
    loss_a = np.zeros((len(train_iterator) * num_epochs,2))
    val_loss_a = np.zeros((len(train_iterator) * num_epochs, 2))
    val_accu_a = np.zeros((len(train_iterator) * num_epochs, 2))
    train_loss_a = np.zeros((len(train_iterator) * num_epochs, 2))
    train_accu_a = np.zeros((len(train_iterator) * num_epochs, 2))
    total_loss = 0
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch + 1} / {num_epochs}]")    
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_model_state(checkpoint)
        
        # model.eval()
        # TODO implement some "simple check of performance"
        model.train()
        optimizer.zero_grad()


        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)
            target_losses = []

            # Forward prop
            # output = model(inp_data, target[:-1, :])
            # print("shape of output {}".format(output.size()))
            # print("shape of output_fc {}".format(output_fc.size()))
            # print("shape of target {}".format(target.size()))
            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            # output = output.reshape(-1)
            # target = target.reshape(-1)
            if batch_idx % optimizer_step == optimizer_step -1:
                optimizer.step()
                optimizer.zero_grad()


            # output, output_fc, decoder_outputs, criterion, losses, total_loss, writer, step = model(inp_data, target, criterion, losses, total_loss, writer, step)
            output, output_fc, decoder_outputs = model(inp_data, target)
            # external loss/backprop
            target_i = target[:, 1:]
            output_fc = torch.flatten(output_fc, start_dim=0, end_dim=1).to(device)
            target_i = torch.flatten(target_i).to(device)
            output_fc = output_fc.float()
            loss = criterion(output_fc, target_i)
            losses.append(loss.item())
            total_loss += loss.item()
            # Back prop
            loss.backward()
            # external loss/backprop end
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            train_batch_correct_sentences = torch.sum(torch.all(torch.eq(target, output), dim=1))
            # Gradient descent step
            # optimizer.step()
            if batch_idx % print_every == print_every - 1:
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_iterator) * num_epochs} \t '
                    f'Train Loss: {total_loss / print_every}')
                
                # do validation control!
                # !important -> we are not yet done with training!
                model.eval()
                val_losses = []
                val_total_loss = 0
                correct_sentences = 0
                total_sentences = 0
                total_batches = 0
                with torch.no_grad():
                    # validate on validation data -> not requested -> plot to see when starting to overfit
                    for v_batch_idx, v_batch in enumerate(train_iterator):
                        v_inp_data = v_batch.src.to(device)
                        v_target = v_batch.trg.to(device)
                        # v_output, v_output_fc, v_decoder_outputs, criterion, val_losses, val_total_loss, writer, step = model(v_inp_data, v_target, criterion, val_losses, val_total_loss, writer, step)
                        v_output, v_output_fc, v_decoder_outputs = model(v_inp_data, v_target)
                        # external loss/backprop
                        v_target_i = v_target[:, 1:] # adjust for "clipped <sos>"                    
                        v_output_fc = torch.flatten(v_output_fc, start_dim=0, end_dim=1).to(device)

                        v_target_i = torch.flatten(v_target_i).to(device)
                        v_output_fc = v_output_fc.float()

                        val_loss = criterion(v_output_fc, v_target_i)
                        val_losses.append(val_loss.item())
                        val_total_loss += val_loss.item()
                        # end external loss / backprop
                        # print("v_target dimensions {}".format(v_target.shape))
                        # print(v_target)
                        # print("v_output dimensions {}".format(v_output.shape))
                        # print(v_output)
                        correct_sentences += torch.sum(torch.all(torch.eq(v_target, v_output), dim=1))
                        total_sentences += batch_size #v_target_i.nelement()
                        total_batches += 1
                        if total_batches >= 20:
                          for i in range(10):                          
                              print("question: {}".format("".join([vocab_train_x.vocab.itos[j] for j in v_inp_data[i]])))
                              print("expected: {}".format("".join([vocab_train_x.vocab.itos[j] for j in v_target[i]])))                            
                              print("predicted: {}".format("".join([vocab_train_x.vocab.itos[j] for j in v_output[i]])))
                          break # too slow GPU to wait for whole training data to process ...
                
                    
                print('\n\n\n')
                print('*' * 30)
                print('*' * 30)
                print('total correct elements {}'.format(correct_sentences))
                print('total elements {}'.format(total_sentences))
                print('*' * 30)
                last_validation_accuracy = correct_sentences / total_sentences
                print('Accuracy: {} %'.format(100 * (correct_sentences / total_sentences)))
                print(f'Validation error rate: {100 - 100 * (correct_sentences / total_sentences): .2f} %')
                print('Validation total loss: {}'.format(val_total_loss / total_batches))
                tr_accu = train_batch_correct_sentences / batch_size
                val_loss_a[step,:] = step, (val_total_loss / total_batches)
                val_accu_a[step,:] = step, last_validation_accuracy
                train_loss_a[step,:] = step, (total_loss / print_every)
                train_accu_a[step,:] = step, tr_accu


                total_loss = 0
                if last_validation_accuracy > 0.98:
                    break       
                
                model.train() # reset to train mode after iterating through validation data
            
            # plot to tensorboard
            # writer.add_scalar("Training loss", loss, global_step=step)
            step += 1
            if last_validation_accuracy > 0.98:
                  break
            

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        if last_validation_accuracy > 0.98:
                  break 
    
    # finished epoch
    fix, ax = plt.subplots()
    ax.set_title(f"Training/Validation loss", fontsize=16)
    # val_loss_a[step,:] = 
    # val_accur_a[step,:] =
    # train_loss_a[step,:] 
    # train_accur_a[step,:]
    plt.xlabel("x", fontsize=10)
    plt.ylabel("loss", fontsize=10)
    tl = ax.plot(train_loss_a[:,0], train_loss_a[:, 1], ".")
    vl = ax.plot(val_loss_a[:, 0], val_loss_a[:, 1], "+")

    fixA, axA = plt.subplots()
    axA.set_title(f"Training/Validation accuracy", fontsize=16)
    # val_loss_a[step,:] = 
    # val_accu_a[step,:] =
    # train_loss_a[step,:] 
    # train_accu_a[step,:]
    plt.xlabel("x", fontsize=10)
    plt.ylabel("accuracy", fontsize=10)
    tlA = axA.plot(train_accu_a[:,0], train_accu_a[:, 1], ".")
    vlA = axA.plot(val_accu_a[:, 0], val_accu_a[:, 1], "+")    
    axA.legend((tlA, vlA), ('training accuracy', 'validation accuracy'), loc="upper right")
    plt.show()
    return model

    


def test_iteration(train_iterator, vocab_train_x, vocab_train_y):
    indexer = 0
    for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            x = batch.src.detach().cpu().numpy()
            y = batch.trg.detach().cpu().numpy()
            x_first = x[1,:]
            y_first = y[1,:]
            print(x_first)
            print(y_first)
            print("question:{}".format("".join([vocab_train_x.vocab.itos[i] for i in x_first])))
            print("answer:{}".format("".join([vocab_train_y.vocab.itos[i] for i in y_first])))
            # for i in range(x.shape[-1]):
            #     print(vocab_train_x.vocab.itos[x[:,i]])
            
            # for j in range(y.shape[-1]):
            #     print(vocab_train_y.vocab.itos[y[:,j]])
            # print([vocab_train_x.vocab.itos[idx]  for idx in x])
            # print([vocab_train_x.vocab.itos[idx]  for idx in y])
            # print(vocab_train_y.vocab.itos(batch.trg))
            indexer += 1
            if indexer >= 10: break


def model_predictions(model, train_iter, vocab_train_x, batch_size):
    model.eval()
    with torch.no_grad():
        for v_batch_idx, v_batch in enumerate(train_iter):
            v_inp_data = v_batch.src.to(device)
            v_target = v_batch.trg.to(device)
            # v_output, v_output_fc, v_decoder_outputs, criterion, val_losses, val_total_loss, writer, step = model(v_inp_data, v_target, criterion, val_losses, val_total_loss, writer, step)
            v_output, v_output_fc, v_decoder_outputs = model(v_inp_data, v_target)            
            for i in range(10):                          
                print("question: {}".format("".join([vocab_train_x.vocab.itos[j] for j in v_inp_data[i]])))
                print("expected: {}".format("".join([vocab_train_x.vocab.itos[j] for j in v_target[i]])))                            
                print("predicted: {}".format("".join([vocab_train_x.vocab.itos[j] for j in v_output[i]])))
            
            break # too slow GPU to wait for whole training data to process ...


def main():
    # data_files_list = get_files_paths()
    # file_statistics(data_files_list)
    # cur_dir = os.path.dirname(__file__)
    data_file_folder = "numbers__place_value"
    batch_size = 64
    # datasource_folder_path = os.path.join(cur_dir, data_file_folder)
    # vocab_train_x, vocab_train_y, train_iter, val_iter = Transformer_dataloader(datasource_folder_path, batch_size=32)    
    vocab_train_x, train_iter, val_iter = Transformer_dataloader('', batch_size=batch_size)    
    # test_iteration(train_iter, vocab_train_x, vocab_train_x)
    trained_model = create_train_transformer(vocab_train_x, train_iter, val_iter, batch_size, 2)
    model_predictions(trained_model, train_iter, vocab_train_x, batch_size)


if __name__ == "__main__":
    main()


