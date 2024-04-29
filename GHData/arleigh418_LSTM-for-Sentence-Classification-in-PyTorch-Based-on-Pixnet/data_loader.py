"""""
This method reference : https://github.com/yuchenlin/lstm_sentence_classifier

"""""


# -*- coding: utf-8 -*-
import sys
import torch
import torch.autograd as autograd
import codecs
import random

import pandas as pd
import numpy as np
import re
import jieba

SEED = 1
jieba.load_userdict('new_words.txt')

def prepare_sequence(seq, to_ix, cuda=False):
    seq = str(seq)
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var


def prepare_predict_sequence(seq, to_ix, cuda=False):
    seq_list = seq.split(' ')
    remainderWords = list(filter(lambda a: a in to_ix, seq_list))
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in remainderWords]))
    
    return var

        

        

def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    # print(len(sentences))
    for sent in sentences:
        sent = str(sent)
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)






def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line



def read_data():

    df0 = pd.read_excel('constellation.xlsx',encoding = 'utf-8')
    df1 = pd.read_excel('food.xlsx',encoding = 'utf-8')
    df2 = pd.read_excel('makeup.xlsx',encoding = 'utf-8')
    df3 = pd.read_excel('medic.xlsx',encoding = 'utf-8')
    df4 = pd.read_excel('mombaby.xlsx',encoding = 'utf-8')
    df5 = pd.read_excel('movie.xlsx',encoding = 'utf-8')
    df6 = pd.read_excel('sport.xlsx',encoding = 'utf-8')
    df7 = pd.read_excel('travel.xlsx',encoding = 'utf-8')
    a0 = df0['內容'].values
    a1 = df1['內容'].values
    a2 = df2['內容'].values
    a3 = df3['內容'].values
    a4 = df4['內容'].values
    a5 = df5['內容'].values
    a6 = df6['內容'].values
    a7 = df7['內容'].values



    # train_data = [(sent,0) for sent in a0[:2000]]+[(sent,1) for sent in a1[:2000]]+[(sent,2) for sent in a2[:2000]]+[(sent,3) for sent in a3[:2000]]+[(sent,4) for sent in a4[:2000]]+[(sent,5) for sent in a5[:2400]]+[(sent,6) for sent in a6[:2000]]+[(sent,7) for sent in a7[:2500]]
    # dev_data = [(sent,0) for sent in a0[2000:2100]]+[(sent,1) for sent in a1[2000:2100]]+[(sent,2) for sent in a2[2000:2100]]+[(sent,3) for sent in a3[2000:2100]]+[(sent,4) for sent in a4[2000:2100]]+[(sent,5) for sent in a5[2400:2500]]+[(sent,6) for sent in a6[2000:2100]]+[(sent,7) for sent in a7[2500:2600]]
    # test_data = [(sent,0) for sent in a0[2100:2500]]+[(sent,1) for sent in a1[2100:2500]]+[(sent,2) for sent in a2[2100:2500]]+[(sent,3) for sent in a3[2100:2500]]+[(sent,4) for sent in a4[2100:2500]]+[(sent,5) for sent in a5[2500:2800]]+[(sent,6) for sent in a6[2100:2500]]+[(sent,7) for sent in a7[2600:3000]]
    train_data = [(sent,0) for sent in a0[:3000]]+[(sent,1) for sent in a1[:3000]]+[(sent,2) for sent in a2[:3000]]+[(sent,3) for sent in a3[:3000]]+[(sent,4) for sent in a4[:3000]]+[(sent,5) for sent in a5[:3400]]+[(sent,6) for sent in a6[:3000]]+[(sent,7) for sent in a7[:3500]]
    dev_data = [(sent,0) for sent in a0[3000:3100]]+[(sent,1) for sent in a1[3000:3100]]+[(sent,2) for sent in a2[3000:3100]]+[(sent,3) for sent in a3[3000:3100]]+[(sent,4) for sent in a4[3000:3100]]+[(sent,5) for sent in a5[3400:3500]]+[(sent,6) for sent in a6[3000:3100]]+[(sent,7) for sent in a7[3500:3600]]
    test_data = [(sent,0) for sent in a0[3100:3500]]+[(sent,1) for sent in a1[3100:3500]]+[(sent,2) for sent in a2[3100:3500]]+[(sent,3) for sent in a3[3100:3500]]+[(sent,4) for sent in a4[3100:3500]]+[(sent,5) for sent in a5[3500:3800]]+[(sent,6) for sent in a6[3100:3500]]+[(sent,7) for sent in a7[3600:4000]]


    
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    label_to_ix = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}
    # len(y)
    # len(label_to_ix)
    
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    return train_data,dev_data,test_data,word_to_ix,label_to_ix



def predict_data(input_path, debug=True):
    df0 = pd.read_excel('constellation.xlsx',encoding = 'utf-8')
    df1 = pd.read_excel('food.xlsx',encoding = 'utf-8')
    df2 = pd.read_excel('makeup.xlsx',encoding = 'utf-8')
    df3 = pd.read_excel('medic.xlsx',encoding = 'utf-8')
    df4 = pd.read_excel('mombaby.xlsx',encoding = 'utf-8')
    df5 = pd.read_excel('movie.xlsx',encoding = 'utf-8')
    df6 = pd.read_excel('sport.xlsx',encoding = 'utf-8')
    df7 = pd.read_excel('travel.xlsx',encoding = 'utf-8')
    a0 = df0['內容'].values
    a1 = df1['內容'].values
    a2 = df2['內容'].values
    a3 = df3['內容'].values
    a4 = df4['內容'].values
    a5 = df5['內容'].values
    a6 = df6['內容'].values
    a7 = df7['內容'].values
    # train_data = [(sent,0) for sent in a0[:2000]]+[(sent,1) for sent in a1[:2000]]+[(sent,2) for sent in a2[:2000]]+[(sent,3) for sent in a3[:2000]]+[(sent,4) for sent in a4[:2000]]+[(sent,5) for sent in a5[:2400]]+[(sent,6) for sent in a6[:2000]]+[(sent,7) for sent in a7[:2500]]
    # dev_data = [(sent,0) for sent in a0[2000:2100]]+[(sent,1) for sent in a1[2000:2100]]+[(sent,2) for sent in a2[2000:2100]]+[(sent,3) for sent in a3[2000:2100]]+[(sent,4) for sent in a4[2000:2100]]+[(sent,5) for sent in a5[2400:2500]]+[(sent,6) for sent in a6[2000:2100]]+[(sent,7) for sent in a7[2500:2600]]
    # test_data = [(sent,0) for sent in a0[2100:2500]]+[(sent,1) for sent in a1[2100:2500]]+[(sent,2) for sent in a2[2100:2500]]+[(sent,3) for sent in a3[2100:2500]]+[(sent,4) for sent in a4[2100:2500]]+[(sent,5) for sent in a5[2500:2800]]+[(sent,6) for sent in a6[2100:2500]]+[(sent,7) for sent in a7[2600:3000]]
    
    train_data = [(sent,0) for sent in a0[:3000]]+[(sent,1) for sent in a1[:3000]]+[(sent,2) for sent in a2[:3000]]+[(sent,3) for sent in a3[:3000]]+[(sent,4) for sent in a4[:3000]]+[(sent,5) for sent in a5[:3400]]+[(sent,6) for sent in a6[:3000]]+[(sent,7) for sent in a7[:3500]]
    dev_data = [(sent,0) for sent in a0[3000:3100]]+[(sent,1) for sent in a1[3000:3100]]+[(sent,2) for sent in a2[3000:3100]]+[(sent,3) for sent in a3[3000:3100]]+[(sent,4) for sent in a4[3000:3100]]+[(sent,5) for sent in a5[3400:3500]]+[(sent,6) for sent in a6[3000:3100]]+[(sent,7) for sent in a7[3500:3600]]
    test_data = [(sent,0) for sent in a0[3100:3500]]+[(sent,1) for sent in a1[3100:3500]]+[(sent,2) for sent in a2[3100:3500]]+[(sent,3) for sent in a3[3100:3500]]+[(sent,4) for sent in a4[3100:3500]]+[(sent,5) for sent in a5[3500:3800]]+[(sent,6) for sent in a6[3100:3500]]+[(sent,7) for sent in a7[3600:4000]]
    
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    label_to_ix = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}
    # len(y)
    # len(label_to_ix)
    df = pd.read_excel(input_path,encoding = 'utf-8')
    df1 = df['內容']
    # df2 = df["class"]
    data = df1.values
  
    text_re=[]
    for i in range(len(data)):
        text_re.append(remove_punctuation(data[i]))
    data_1 = [' '.join(jieba.cut(d)) for d in text_re]
    
    data_2 = [sent for sent in data_1]

    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    return data_2,word_to_ix
