from __future__ import print_function
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from read_user import getData
import pickle

start = time.time()

# Constants
train_set_num = 100000
test_set_num = 10000
hidden_size = 64
batch_size = 1
use_cuda = torch.cuda.is_available()
personality_min = 12
personality_max = 60
word_embedding_size = 128
personality_dimension_size = 5

print('cuda available:', use_cuda)

if use_cuda:
    torch.cuda.manual_seed_all(2222)
    print('cuda', torch.cuda.current_device())
if torch.has_cudnn:
    print('has cudnn')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def toLabels(labels, user_id):
    # print('take user ',user_id,' label')
    return labels[user_id]

def splitTestSet(all_user_data, train, test):
    train = int(len(all_user_data) * train)
    test = int(len(all_user_data) * test)
    train_set = []
    test_set = []
    for i, d in enumerate(all_user_data):
        if i < train:
            train_set.append(d)
        else:
            test_set.append(d)
    return train_set, test_set



all_user_data, user_labels, vocab, users, max_sentence_len = getData()
vocabulary_pickle = open('vocabulary.pickle', 'wb')
pickle.dump(vocab, vocabulary_pickle, protocol=pickle.HIGHEST_PROTOCOL)
print('save vocab to vocabulary.pickle')

from Conv_Rnn_Classifier import Net
net = Net(hidden_size=hidden_size, batch_size=batch_size, use_cuda=use_cuda, vocabulary_size=len(vocab), embedding_size=word_embedding_size, output_size=personality_dimension_size, max_sentence_len=max_sentence_len)
if use_cuda: net.cuda()

import torch.optim as optim
criterion = nn.MSELoss()

if use_cuda: criterion.cuda()
optimizer = optim.Adam(net.parameters())

import random
random.shuffle(all_user_data)


train_set, test_set = splitTestSet(all_user_data, train=0.8, test=0.2)


print('start training')
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_set, 0):

        user_id, sentence_tensor = data.items()[0]

        sentence_tensor = Variable(sentence_tensor)
        sentence_tensor = sentence_tensor.cuda() if use_cuda else sentence_tensor

        #zero the parameter gradients
        optimizer.zero_grad()

        output = net(sentence_tensor)

        ready_label = toLabels(user_labels, user_id)
        ready_label = ready_label.cuda() if use_cuda else ready_label

        loss = criterion(output, ready_label)

        loss.backward()

        optimizer.step()

        #percentage loss
        running_loss += loss.data[0] / (5 * 1)
        if i % 50 == 0 and i != 0:
            print('[%d, %5d] loss:%.5f' %
                  (epoch, i, running_loss/50))
            running_loss = 0.0
print('finished training')


print('start testing')
test_running_loss = 0
total = 0
for i, data in enumerate(test_set, 0):
    user_id, sentence_tensor = data.items()[0]

    sentence_tensor = Variable(sentence_tensor)
    sentence_tensor = sentence_tensor.cuda() if use_cuda else sentence_tensor

    output = net(sentence_tensor)

    ready_label = toLabels(user_labels, user_id)
    ready_label = ready_label.cuda() if use_cuda else ready_label

    loss = criterion(output, ready_label)

    # percentage loss
    test_running_loss += loss.data[0] / (5 * 1)
    total += 1

print('###average loss on test set###:', test_running_loss/total)
print('finished testing')
torch.save(net.state_dict(), "text_personality_classifier.dict_state")
print('model saved')

end = time.time()
print('time elapsed', end-start)


