# 词向量由单词前后的单词来定义
# 用text8.train来训练词向量
# 用其他文件来评测生成的词向量
# 用datasets来获取和处理文件
# 用dataloader来获取测试数据
# 模型用Embedding函数来处理输入输出数据
# 用SGD做梯度下降



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

SEED = 24
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)
    
# 设定一些超参数
    
K = 100 # number of negative samples
C = 3 # nearby words threshold
NUM_EPOCHS = 2 # The number of epochs of training
MAX_VOCAB_SIZE = 20000 # the vocabulary size
BATCH_SIZE = 32 # the batch size
LEARNING_RATE = 0.2 # the initial learning rate
EMBEDDING_SIZE = 100
       
    
LOG_FILE = "word_embedding.log"

# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()

with open(".data\\text8\\text8.train.txt", "r") as fin:
    text = fin.read()
    
# 处理文本，并且设置单词与下标之间的map
text = [w for w in word_tokenize(text.lower())]
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()] 
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs) # 用来做 negative sampling
VOCAB_SIZE = len(idx_to_word)

# 建立标准的dataset来处理text
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE-1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)
        
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices] 
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        
        return center_word, pos_words, neg_words 

dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
# 用dataloader来负责将词语分块以及生成数据的迭代器
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Windows不支持？显卡不支持？

# 建立模型，由nn.Embedding组成
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        # 用uniform_来初始化weight
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        
        
    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        
        return: loss, [batch_size]
        '''
        
        batch_size = input_labels.size(0)
        
        # 将单词的下标转换成对应的词向量
        input_embedding = self.in_embed(input_labels) # B * embed_size
        pos_embedding = self.out_embed(pos_labels) # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels) # B * (2*C * K) * embed_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze() # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze() # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1) # batch_size

        loss = log_pos + log_neg
        
        return -loss
    
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()
        
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
model = model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        
        # TODO
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)
            
        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
            
        '''
        if i % 2000 == 0:
            embedding_weights = model.input_embeddings()
            sim_simlex = evaluate(".data\\text8\\simlex-999.txt", embedding_weights)
            sim_men = evaluate(".data\\text8\\men.txt", embedding_weights)
            sim_353 = evaluate(".data\\text8\\wordsim353.csv", embedding_weights)
            with open(LOG_FILE, "a") as fout:
                print("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                    e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
                fout.write("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                    e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
        '''    
        if i == 2000:
            break    
    embedding_weights = model.input_embeddings()
    np.save("word_embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), "word_embedding-{}.th".format(EMBEDDING_SIZE))

'''
def evaluate(filename, embedding_weights): 
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))

    return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity

def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]
'''
'''
model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))
embedding_weights = model.input_embeddings()
print("simlex-999", evaluate(".data\\text8\\simlex-999.txt", embedding_weights))
print("men", evaluate(".data\\text8\\men.txt", embedding_weights))
print("wordsim353", evaluate(".data\\text8\\wordsim353.csv", embedding_weights))

for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word))

man_idx = word_to_idx["man"] 
king_idx = word_to_idx["king"] 
woman_idx = word_to_idx["woman"]
embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
for i in cos_dis.argsort()[:20]:
    print(idx_to_word[i])
'''