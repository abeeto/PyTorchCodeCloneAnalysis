dependencies = ['torch', 'transformers', 'os']
from models.transformer_cnn import *
from models.transformer_cnn_no_stats import *
import os
import torch
from transformers import AutoModel

model_urls = {"with stats": "https://github.com/TanveerMittal/BERT-Feature-Type-Inference/releases/download/capstone/BERT_fti_with_stats.pt",
              "no stats": "https://github.com/TanveerMittal/BERT-Feature-Type-Inference/releases/download/capstone/BERT_fti_no_stats.pt"}

def BERT_fti_with_stats(pretrained=True, **kwargs):
    # load pretrained bert_model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # freeze bert weights
    for param in bert.parameters():
        param.requires_grad = False

    # initialize model
    model = transformer_cnn(bert, 256, [1, 2, 3, 5])

    # load pretrained cnn weights
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls["with stats"]))

    return model

def BERT_fti_no_stats(pretrained=True, **kwargs):
    # load pretrained bert_model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # freeze bert weights
    for param in bert.parameters():
        param.requires_grad = False

    # initialize model
    model = transformer_cnn_no_stats(bert, 256, [1, 2, 3, 5])

    # load pretrained cnn weights
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls["no stats"]))

    return model