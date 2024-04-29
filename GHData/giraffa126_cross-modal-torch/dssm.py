import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSSM(nn.Module):
    def __init__(self, vocab_size=500000, embed_size=256, 
            hidden_size=128):
        super(DSSM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.query_fc = nn.Linear(embed_size, hidden_size)
        nn.init.xavier_uniform_(self.query_fc.weight)
        nn.init.constant_(self.query_fc.bias, 1e-8)

        self.title_fc = nn.Linear(embed_size, hidden_size)
        nn.init.xavier_uniform_(self.title_fc.weight)
        nn.init.constant_(self.title_fc.bias, 1e-8)

        self.tanh = nn.Tanh()
        
    def forward(self, query, pos_title, neg_title):
        query_embed = self.embedding(query)
        query_embed_pool = torch.sum(query_embed, 1)

        pos_title_embed = self.embedding(pos_title)
        pos_title_embed_pool = torch.sum(pos_title_embed, 1)

        neg_title_embed = self.embedding(neg_title)
        neg_title_embed_pool = torch.sum(neg_title_embed, 1)

        query_vec = self.query_fc(query_embed_pool)
        query_vec = self.tanh(query_vec)

        pos_title_vec = self.title_fc(pos_title_embed_pool)
        pos_title_vec = self.tanh(pos_title_vec)

        neg_title_vec = self.title_fc(neg_title_embed_pool)
        neg_title_vec = self.tanh(neg_title_vec)
        
        left = torch.cosine_similarity(query_vec, pos_title_vec, 
                dim=1, eps=1e-8)
        right = torch.cosine_similarity(query_vec, neg_title_vec, 
                dim=1, eps=1e-8)
        return left, right

    def predict(self, query):
        query_embed = self.embedding(query)
        query_embed_pool = torch.sum(query_embed, 1)
        query_vec = self.query_fc(query_embed_pool)
        query_vec = self.tanh(query_vec)
        return query_vec

    def calc_sim(self, query, pos_title):
        query_embed = self.embedding(query)
        query_embed_pool = torch.sum(query_embed, 1)
        query_vec = torch.tanh(self.query_fc(query_embed_pool))

        pos_title_embed = self.embedding(pos_title)
        pos_title_embed_pool = torch.sum(pos_title_embed, 1)
        pos_title_vec = torch.tanh(self.title_fc(pos_title_embed_pool))
        return torch.cosine_similarity(query_vec, pos_title_vec, dim=1, eps=1e-8)
        

class RankLoss(nn.Module):
    def __init__(self, enlarge=5.0):
        super().__init__()
        self.enlarge = enlarge

    def forward(self, left, right):
        diff = (left - right) * self.enlarge
        loss = torch.log1p(diff.exp()) - diff
        return torch.mean(loss)

