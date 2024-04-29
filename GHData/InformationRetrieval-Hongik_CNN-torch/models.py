import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
     def __init__(self, vocab_size=10002, embedding_dim=128, n_filters = 100, filter_sizes = [3, 4, 5], output_dim = 1, dropout = 0.5):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim)) for fs in filter_sizes
                                    ])
        self.fully_Connected = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
     def forward(self, x):
        # x = [batch_size, maxLen]

        embedding_vec = self.embedding(x)
        # embedding_vec = [batch_size, maxLen, emb_dim]
        
        embedding_vec = embedding_vec.unsqueeze(1)
        # # embedding_vec = [batch_size, 1, maxLen, emb_dim]
        
        conv_vecs = [F.relu(conv(embedding_vec)).squeeze(3) for conv in self.convs]

        # conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1]

        pooled_vecs = [F.max_pool1d(conv_vec, conv_vec.shape[2]).squeeze(2) for conv_vec in conv_vecs]
        # pooled_vec = [batch_size, n_filters]

        concat = self.dropout(torch.cat(pooled_vecs, dim = 1))
        # concat = [batch_size, n_filters * len(filter_sizes)]

        out = self.sigmoid(self.fully_Connected(concat))
        # out = [batch_size, 1]

        return out 

class Double_CNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=100, vector_len = 80, n_filters_a = 64, n_filters_b = 128, filter_sizes_a = [3, 4, 5], filter_sizes_b = [3, 4, 5], dropout = 0.5):
        super(Double_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_list_a = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters_a,
                                              kernel_size = (fs_a, embedding_dim)) for fs_a in filter_sizes_a
                                    ])
        
        self.conv_list_b = nn.ModuleList([
                                    nn.Conv2d(in_channels = n_filters_a,
                                              out_channels = n_filters_b,
                                              kernel_size = (fs_b, 1)) for fs_b in filter_sizes_b
                                    ])
        
        self.fully_Connected = nn.Linear(len(filter_sizes_b) * n_filters_b, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [batch_size, maxLen]
        
        embedding_vec = self.embedding(x)
        # embedding_vec = [batch_size, maxLen, emb_dim]
        embedding_vec = embedding_vec.unsqueeze(1)
        # embedding_vec = embedding_vec.permute(0, 2, 1)
        # # embedding_vec = [batch_size, 1, maxLen, emb_dim]
        
        conv_vecs = [self.relu(conv(embedding_vec)) for conv in self.conv_list_a]
        # conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1, 1]
        # for conv in self.conv_vecs:
        #     print(conv.shape + "\n")
            
        conv_vecs = [self.relu(self.conv_list_b[idx](conv_vecs[idx])).squeeze(3) for idx in range(len(self.conv_list_b))]
        # conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1]
        
        pooled_vecs = [F.max_pool1d(conv_vec, conv_vec.shape[2]).squeeze(2) for conv_vec in conv_vecs]
        # pooled_vec = [batch_size, n_filters]
        
        concat = self.dropout(torch.cat(pooled_vecs, dim = 1))
        # concat = [batch_size, n_filters * len(filter_sizes)]
        
        out = self.sigmoid(self.fully_Connected(concat))
        # out = [batch_size, 1]
        
        return out