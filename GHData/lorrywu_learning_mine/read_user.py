from os import walk
import json
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(dirpath + "/" + f)
        break
    return files

def word2vocab(vocab, w):
    index = len(vocab)
    if w not in vocab:
        vocab[w] = index
        print 'add', w



def wordlist2vocab(vocab, wordlist):
    for w in wordlist:
        word2vocab(vocab, w)

def read(path='seg_char_data'):
    max_sentence_len = 0
    files = exploreFolder(path)
    vocab = {}
    all_user = []
    for f in files:
        user_id = f.split('/')[1].split('.')[0]
        user_id = int(user_id)
        for line in open(f, 'r'):
            sentence_list = json.loads(line)
            wordlist2vocab(vocab, sentence_list)
            if len(sentence_list) > max_sentence_len:
                max_sentence_len = len(sentence_list)
            all_user.append({user_id:sentence_list})
    return all_user, vocab, max_sentence_len


def sentence2tensor(all_user, vocab):
    new_all_user = []
    for data in all_user:
        user_id, sentence = data.items()[0]
        sentence_tensor = torch.LongTensor([[[vocab[word]] for word in sentence]])
        new_all_user.append({user_id:sentence_tensor})
    return new_all_user, vocab

def read_users(path='data/users'):
    users = []
    for i in open(path, 'r'):
        users.append(int(i))
    return users

def normalization(d, max = 60, min = 12):
    return (d - min)/(max - min)

def read_labels(users, path='labels/cleaned_label2.csv'):
    csv = pd.read_csv(path, index_col='user_id', sep=';')
    csv = csv.drop_duplicates()
    all_labels = {}
    for u in users:
        try:
            csv.loc[u]
        except:
            print 'key error', u
            continue

        ready_label = []
        ready_label.append(normalization(float(csv.loc[u]['Neuroticism'])))
        ready_label.append(normalization(float(csv.loc[u]['Extraversion'])))
        ready_label.append(normalization(float(csv.loc[u]['Openness'])))
        ready_label.append(normalization(float(csv.loc[u]['Agreeableness'])))
        ready_label.append(normalization(float(csv.loc[u]['Conscientiousness'])))

        # ready_label.append(float(csv.loc[u]['Neuroticism']))
        # ready_label.append(float(csv.loc[u]['Extraversion']))
        # ready_label.append(float(csv.loc[u]['Openness']))
        # ready_label.append(float(csv.loc[u]['Agreeableness']))
        # ready_label.append(float(csv.loc[u]['Conscientiousness']))

        ready_label = [ready_label]
        tensor = torch.from_numpy(np.array(ready_label)).float()


        all_labels[u] = Variable(tensor)

    return all_labels


def getData():
    users = read_users()
    labels = read_labels(users)
    all_user_data, vocab, max_sentence_len = read()
    all_user_data, vocab = sentence2tensor(all_user_data, vocab)
    return all_user_data, labels, vocab, users, max_sentence_len

# users = read_users(path='data/users')
# labels = read_labels(users)
# print labels[1988788063]
