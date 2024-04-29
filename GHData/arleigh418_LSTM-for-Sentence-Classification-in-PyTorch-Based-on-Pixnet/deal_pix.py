# -*- coding: UTF-8 -*-


import json
import numpy as np
from bs4 import BeautifulSoup
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
jieba.load_userdict('new_words.txt')

data = []
with open('D:/2017-pixnet-hackathon-data/article/food.json',encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

b = [] 
for n in range(len(data)):
    try:
        b.append(data[n]['content'])
    except KeyError:
        continue

text = []
for i in range(len(b)):
    text.append(BeautifulSoup(b[i]).text.split('\n'))

all_text=[]
for i in range(len(text)):
    all_text.append(''.join(text[i])+str(']'))

# print(all_text)
# print('=================one==================')

def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line

text_re=[]
for i in range(len(all_text)):
    text_re.append(remove_punctuation(all_text[i]))

words = [' '.join(jieba.cut(d)) for d in text_re]

df = pd.DataFrame(text_re)
df.to_excel('food.xlsx',encoding='UTF-8')

