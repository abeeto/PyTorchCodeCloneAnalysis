import sys
import os
import torch
import torch.nn as nn
from dssm import DSSM
import torchvision.models as models

class CrossModal(nn.Module):
    def __init__(self, vocab_size=250000, 
            embed_size=128, hidden_size=512,
            pretrain_path=None):
        super(CrossModal, self).__init__()
        # image
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.resnet_linear = nn.Linear(resnet.fc.in_features, 
            hidden_size)
        self.resnet_bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

        # text
        self.dssm = DSSM(vocab_size=vocab_size) 
        self.dssm.load_state_dict(torch.load(pretrain_path))
        self.dssm_linear = nn.Linear(embed_size, hidden_size)
        self.dssm_bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

        # Function
        self.tanh = nn.Tanh()

    def forward(self, query, pos_img, neg_img):
        #with torch.no_grad():
        text_feature = self.dssm.predict(query)
        pos_img_feature = self.resnet(pos_img)
        neg_img_feature = self.resnet(neg_img)

        text_feature = self.tanh(
            self.dssm_bn(
            self.dssm_linear(
                text_feature
        )))
        
        pos_img_feature = pos_img_feature.reshape(pos_img_feature.size(0), -1)
        pos_img_feature = self.tanh(self.resnet_bn(self.resnet_linear(pos_img_feature)))

        neg_img_feature = neg_img_feature.reshape(neg_img_feature.size(0), -1)
        neg_img_feature = self.tanh(self.resnet_bn(self.resnet_linear(neg_img_feature)))

        left = torch.cosine_similarity(text_feature, pos_img_feature)
        right = torch.cosine_similarity(text_feature, neg_img_feature)
        return left, right

    def query_emb(self, query):
        text_feature = self.dssm.predict(query)
        text_feature = self.tanh(
            self.dssm_bn(
            self.dssm_linear(
                text_feature
        )))
        return text_feature

    def img_emb(self, pos_img):
        pos_img_feature = self.resnet(pos_img)
        pos_img_feature = pos_img_feature.reshape(pos_img_feature.size(0), -1)
        pos_img_feature = self.tanh(self.resnet_bn(self.resnet_linear(pos_img_feature)))
        return pos_img_feature


class RankLoss(nn.Module):
    def __init__(self, enlarge=5.0):
        super().__init__()
        self.enlarge = enlarge

    def forward(self, left, right):
        diff = (left - right) * self.enlarge
        loss = torch.log1p(diff.exp()) - diff
        return torch.mean(loss)

