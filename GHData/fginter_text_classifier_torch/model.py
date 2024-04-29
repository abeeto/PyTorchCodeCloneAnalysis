import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import config
import sys

class TClass(nn.Module):

    def __init__(self,class_count,wrd_emb_dims,chr_emb_dims,lstm_size=30):
        super(TClass, self).__init__()
        wrd_e_num,wrd_e_dim=wrd_emb_dims
        chr_r_num,chr_e_dim=chr_emb_dims
        self.chr_embedding=nn.Embedding(num_embeddings=wrd_e_num,embedding_dim=wrd_e_dim,padding_idx=0,sparse=True)
        self.wrd_embedding=nn.Embedding(num_embeddings=wrd_e_num,embedding_dim=wrd_e_dim,padding_idx=0,sparse=True)

        self.wrd_bilstm=nn.LSTM(input_size=self.wrd_embedding.embedding_dim,hidden_size=lstm_size,num_layers=1,bidirectional=True)

        self.chr_bilstm=nn.LSTM(input_size=self.chr_embedding.embedding_dim,hidden_size=wrd_e_dim//2,num_layers=1,bidirectional=True)
        self.chr2wrd_bilstm=nn.LSTM(input_size=self.chr_bilstm.hidden_size*self.chr_bilstm.num_layers*2,hidden_size=lstm_size,num_layers=1,bidirectional=True)

        self.dense1=nn.Linear(in_features=self.wrd_bilstm.hidden_size*self.wrd_bilstm.num_layers*2,out_features=100)
        self.dense2=nn.Linear(in_features=self.dense1.out_features, out_features=class_count)

    def forward(self,minibatch_wrd,minibatch_wrd_lengths,minibatch_chr,minibatch_chr_lengths):
        # First, let us run character-based LSTM on the words
        # minibatch_chr is word_count x minibatch x char_count
        # for the char-lstm we need to get it to char_count x word
        wcount,mbatch,ccount=minibatch_chr.size()  # word X sentence X character

        # character x word
        char_lstm_in=minibatch_chr.contiguous().view(wcount*mbatch,ccount).transpose(0,1).contiguous()
        # length
        char_lstm_in_lens=minibatch_chr_lengths.contiguous().view(wcount*mbatch)
        # length                 index
        char_lstm_in_lens_sorted,char_lstm_in_lens_sorted_idx=torch.sort(char_lstm_in_lens,descending=True)
        #print("char_lstm_in_lens_sorted",char_lstm_in_lens_sorted,"char_lstm_in_lens_sorted_idx",char_lstm_in_lens_sorted_idx)

        # number scalar
        nonzero_words_count=char_lstm_in_lens_sorted.squeeze().nonzero().size()[0]

        # character x word  ... sorted from longest to shortest word, only nonzero length
        char_lstm_in_sorted=char_lstm_in[:,char_lstm_in_lens_sorted_idx[:nonzero_words_count]]
        # length ... only nonzero
        char_lstm_in_lens_sorted=char_lstm_in_lens_sorted.squeeze()[:nonzero_words_count]
        # character x word x embeddingdim
        char_lstm_in_sorted_embedded=self.chr_embedding(char_lstm_in_sorted)
        print("input",char_lstm_in_sorted_embedded,"lengths",char_lstm_in_lens_sorted.squeeze())
        char_lstm_in_sorted_packed=rnn.pack_padded_sequence(input=char_lstm_in_sorted_embedded,lengths=list(char_lstm_in_lens_sorted.data))
        print("char_lstm_in_sorted_packed",char_lstm_in_sorted_packed)
        
        _,(chr_h_n,_)=self.chr_bilstm(char_lstm_in_sorted_packed)
        print("chr_h_n",chr_h_n)
        #print("char_h_n.size()",char_h_n.size())
        #[2, 45000, 30]
        #  (2,45000,30)  -->  (45000,2,30)
        chr_h_n_wrd_input=chr_h_n.transpose(0,1).contiguous().view(wcount,mbatch,-1)
        #print("chr_h_n_wrd_input.size()",chr_h_n_wrd_input.size())


        minibatch_wrd_emb=self.wrd_embedding(minibatch_wrd)
        #print("minibatch_wrd_emb.size()",minibatch_wrd_emb.size())

        chr_h_n_wrd_input_sum=(chr_h_n_wrd_input)#+minibatch_wrd_emb)/
        
        _,(h_n,_)=self.chr2wrd_bilstm(chr_h_n_wrd_input_sum)
        _,batch,_=h_n.size()
        #print("h_n.size()",h_n.size())
        

        #wrd_bilstm_out,(h_n,c_n)=self.wrd_bilstm(minibatch_wrd_emb)
        #layers_dirs,batch,feats=h_n.size()
        #steps,batch,feats=wrd_bilstm_out.size()

        h_n_linin=h_n.transpose(0,1).contiguous().view(batch,-1)
        dense1_out=F.tanh(self.dense1(h_n_linin))
        dense2_out=self.dense2(dense1_out)
        return dense2_out
        

if __name__=="__main__":
    x=TClass(class_count=2)
