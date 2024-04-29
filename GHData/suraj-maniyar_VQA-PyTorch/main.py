import numpy as np
import pickle, os

from model import LanguageModel, VQA_FeatureModel
from data_loader import ImageFeatureDataset
from data_utils import change, preprocess_text
from model import VQA_FeatureModel
from trainer import train

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


config, glove = {}, {}
train_features, val_features = [], []


### Loading GLoVe

#dim = 300
dim = 50
topK = 500  # selects topK frequent words and sets other as <unk> (unknown) 

with open('data/glove.6B.'+str(dim)+'d.txt', 'r') as f: 
    embedding = f.read()

for element in embedding.split('\n'):
    word = element.split(' ')[0]
    vec = [float(temp) for temp in element.split(' ')[1:]]
    glove[word] = np.array(vec)                                                                                                                

glove['<end>']   = np.array(glove['----'])
glove['<unk>']   = np.array(glove['-----'])
glove['<pad>']   = np.array([0.0]*dim) 

print('Loaded GloVe') 


### Loading Image Features

with open('dumps/train_features_vgg16.pkl', 'rb') as f:
    original_features = pickle.load(f)
print('len(original_features) : ', len(original_features))


with open('dumps/val_features_vgg16.pkl', 'rb') as f:
    original_val_features = pickle.load(f)
print('len(original_val_features) : ', len(original_val_features))


for i in range(len(original_features)):
    features, path, question, answer = original_features[i]
    answer = preprocess_text(answer)
    if(len(answer) == 2):
        train_features.append(original_features[i])

for i in range(len(original_val_features)):
    feature, path, question, answer = original_val_features[i]
    answer = preprocess_text(answer)
    if(len(answer) == 2):
        val_features.append(original_val_features[i])


print('len(train_features) : ', len(train_features))
print('len(val_features) : ', len(val_features))

print('\n----------------------------------------\n')


question, answer = [], ['<unk>']
question_len = []

for i in range(len(train_features)):  
    _, _, ques, ans = train_features[i]
    ques = preprocess_text(ques)
    ans = preprocess_text(ans)
    
    if(len(ans) == 2): 
        question.extend(ques)
        question_len.append(len(ques))
        answer.append(ans[0])




# print('MAX Question LEN (INPUT SEQ LENGTH): ', max(question_len))

with open('dumps/word_list.pkl', 'rb') as f:
    word_list = pickle.load(f)

word_list = word_list[0:topK]


for i in range(len(answer)):
    answer[i] = change(answer[i])
    if answer[i] not in word_list:
        answer[i] = '<unk>'


unique_question = list(set(question))
unique_answer = list(set(answer))

unique_question = sorted(unique_question)
unique_answer = sorted(unique_answer)

print('len(unique_question) : ', len(unique_question))
print('len(unique_answer) : ', len(unique_answer))


input_intersection = set(unique_question) & set(glove.keys())
print('len(input_intersection) : ', len(input_intersection))

input_embedding = {}
for key in input_intersection:
    input_embedding[key] = glove[key]

input_embedding['<unk>'] = glove['<unk>']
input_embedding['<end>'] = glove['<end>']
input_embedding['<pad>'] = glove['<pad>']


sorted_keys = sorted(unique_answer)

word2idx, idx2word = {}, {}
for i, word in enumerate(sorted_keys):
    word2idx[word] = i
    idx2word[i] = word



with open('dumps/input_embedding.pkl', 'wb') as f:
    pickle.dump(input_embedding, f)

with open('dumps/idx2word.pkl', 'wb') as f:
    pickle.dump(idx2word, f)

print('\nSAVED\n')

print('Input Embedding : ', len(input_embedding))
print('idx2word : ', len(idx2word))





# Finding frequency of answer words for inverse weights initialization 

answer_train = ['<unk>']
for i in range(len(train_features)):
    ans = train_features[i][3]
    ans = preprocess_text(ans)
    
    if(len(ans) == 2):
        ans = change(ans[0])
        if ans not in word_list:
            ans = '<unk>'
        answer_train.append(ans)


from collections import Counter

counter_train = Counter(answer_train)
keys_train = list(counter_train.keys())
values_train = list(counter_train.values())

freq_train = {}
for i in range(len(keys_train)):
    freq_train[ keys_train[i] ] = values_train[i]


#################################################################################################


config['img_size'] = 224
config['batch_size'] = 128

config['vocab_size'] = len(word2idx)
config['input_seq_len'] = 22 #max(question_len)
config['embedding_size'] = dim

config['num_hidden_units'] = 512
config['num_layers'] = 2

config['dropout'] = 0.4
config['learning_rate'] = 0.001
config['epochs'] = 400

config['image_feature'] = 1024
config['question_feature'] = 1024




#################################################################################################




train_dataset = ImageFeatureDataset(config, train_features, input_embedding, word2idx, word_list)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

val_dataset = ImageFeatureDataset(config, val_features, input_embedding, word2idx, word_list)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])




if os.path.isfile('checkpoint/model_vgg16.pth'):
    model = torch.load('checkpoint/model_vgg16.pth')
    print('\nModel Loaded from Disk\n')
else:
    model = VQA_FeatureModel(config)
    print('\nNo checkpoint found\n')


if torch.cuda.is_available():
    model = model.cuda()


print('\n\n########## MODEL ############\n')
print(model)
print('\n')
print('\n########### CONFIG ############\n')
print(config)
print('\n')
print('########### TRAINING ############\n')





criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


train(config, model, train_loader, val_loader, optimizer, criterion)


