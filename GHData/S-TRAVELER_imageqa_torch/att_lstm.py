import torch
import torch.nn.functional as F
import numpy

from attention import *

nn=torch.nn

class att_lstm(torch.nn.Module):
    def __init__(self, options):
        super(att_lstm,self).__init__()
        self.sent_drop=options['sent_drop']
        self.drop_ratio=options['drop_ratio']

        self.lstm=nn.LSTM(
                    input_size=options['n_emb'],
                    hidden_size=options['rnn_size'],
                    num_layers=1,
                    batch_first=True    # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)     
                )
        self.dropout=nn.Dropout()
        self.image_mlp_act=nn.Sequential(
            nn.Linear(512,1024),
            nn.Tanh(),
            nn.Dropout()
        )
        self.att1=Attention(options)
        self.att2=attention(options)
        self.w_emb= nn.Embedding(options['n_words']+1, options['n_emb'],padding_idx=0,max_norm=1)

        self.combined_mlp_0=nn.Sequential(
            torch.squeeze(dim=1),
            nn.Dropout(),
            nn.Linear(1024,1000),
        )
        self.output=nn.Sequential(
            nn.Softmax(),
            torch.argmax(dim=1)
        )

    def forward(self, image_feat, input_idx):
        input_emb = self.w_emb(input_idx)
        if self.sent_drop:
            input_emb=nn.Dropout(input_emb, self.drop_ratio)
        
        h_encode, (h_n, h_c) = self.rnn(input_emb, None)  
        h_encode=h_encode[:, -1]
        image_feat_down=self.image_mlp_act(image_feat)
        combined_hidden_1=self.att1(image_feat_down,h_encode)
        combined_hidden=self.att2(image_feat_down, combined_hidden_1) #(100,1,1024)
        combined_hidden=self.combined_mlp_0(combined_hidden,dim=1) #(100,1000)
        pred_label = self.output(combined_hidden)
        return pred_label


