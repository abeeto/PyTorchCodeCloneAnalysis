# -*- coding: utf-8 -*-
import torch.nn as nn


class Naive_Clf(nn.Module):
    def __init__(self, vocab_size, dim_embdeding, num_classes=2, fix_length=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embdeding)
        self.classifier = nn.Linear(dim_embdeding*fix_length, 2)
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(dim_embdeding * fix_length, 2),
#                nn.ReLU(),
#                nn.Linear(512, num_classes)
                )        
    def forward(self, x):
#        print(x)
        out = self.embedding(x).view(x.size(0), -1)
#        print(embeds.shape)
        out = self.classifier(out)
        return out
    
class IMDBRnn(nn.Module):
    def __init__(self, vocab_size, dim_embdeding, num_classes=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embdeding)
        h1, h2 =  256, 512
        self.rnn = nn.LSTM(dim_embdeding, h1, num_layers=num_layers, batch_first=True, dropout =0.5)
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(h1, num_classes),
#                nn.Linear(h1, h2),
#                nn.ReLU(),
#                nn.Linear(h2, num_classes)
                )
        
#    In the forward function, we pass the input data of size [200, 32], 
#    which gets passed through the embedding layer and each token in the batch gets replaced by embedding 
#    and the size turns to [200, 32, 100], where 100 is the embedding dimensions.    
     
#    The LSTM layer takes the output of the embedding layer along with two hidden variables. 
#    The hidden variables should be of the same type of the embeddings output, 
#    and their size should be [num_layers, batch_size, hidden_size]. 

#    The LSTM processes the data in a sequence and generates the output of the shape [Sequence_length, batch_size, hidden_size], 
#    where each sequence index represents the output of that sequence. 
#    In this case, we just take the output of the last sequence, which is of shape [batch_size, hidden_dim]. 
        
    def forward(self, x):
        self.rnn.flatten_parameters()

        e_out = self.embedding(x)
        rnn_o, _ = self.rnn(e_out) 
#        print(rnn_o.shape)
#        rnn_o = rnn_o[-1]
        rnn_o = rnn_o[:, -1, :].squeeze()

        return self.classifier(rnn_o)
    
class IMDBCnn(nn.Module):
    def __init__(self, vocab_size, dim_embdeding, num_classes, kernel_size=3, fix_length=200):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_embdeding)        
        out_channels, output_size = 256, 20
        
        self.cnn = nn.Conv1d(in_channels = fix_length, out_channels=out_channels, kernel_size=3)
        self.avg = nn.AdaptiveAvgPool1d(output_size = output_size)
#        self.fc = nn.Linear(256 * 10, num_classes)

        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(out_channels * output_size, num_classes),
#                nn.ReLU(),
#                nn.Linear(512, num_classes)
                )
        
    def forward(self, x):

        e_out = self.embedding(x)
        
        cnn_o = self.cnn(e_out) 
        cnn_avg = self.avg(cnn_o)
        cnn_avg = cnn_avg.view(x.shape[0], -1)
        return self.classifier(cnn_avg)
