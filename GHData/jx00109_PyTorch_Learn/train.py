# -*- encoding:utf8 -*-
import torch
import torch.autograd as autograd
import utils as u
import model

datas = ["I am test sentence one", "You are the one he like", "I like eating"]

mydict = u.getDict(datas)

batch_size = 2
time_step = 5
embed_dim = 3
hidden_dim = 4

sentences = u.load_set(datas, mydict)

paded_sents, masks = u.padding_and_generate_mask(sentences, time_step)


LSTM = model.myLSTM(embed_dim, hidden_dim, len(mydict))
for s, m in u.batch_iter(paded_sents, masks, batch_size):
    s = autograd.Variable(torch.LongTensor(s), requires_grad=False)
    print s
    m = autograd.Variable(torch.FloatTensor(m), requires_grad=False)
    final_h = LSTM(s, m, time_step, batch_size)
    print final_h
    print '-' * 80
