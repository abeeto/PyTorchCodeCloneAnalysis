# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from torchtext import datasets
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--u_sample_ratio', action='store', default=1, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 200
GENERATED_NUM = 10000
# PRE_EPOCH_NUM = 120
PRE_EPOCH_NUM = 50
SEQ_LEN = 100

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 100
g_hidden_dim = 40
# g_sequence_len = 100

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2



def generate_samples(model,h, batch_size, generated_num, SOS):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(h, batch_size, SEQ_LEN, SOS).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for _, data,_ in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data[:,1:])
        h0 = Variable(torch.zeros((1, len(data), g_hidden_dim)))
        if opt.cuda:
            data = data.cuda()
            h0 = h0.cuda()
        pred = model.forward(data,h0)
        loss = criterion(pred[:-1], data.contiguous().view(-1)[1:])
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return math.exp(total_loss / total_words)

def discriminator_train_epoch(model, real_loader, fake_loader, criterion, optimizer):
    total_loss = 0.
    total_words = 0.

    fake_iter = iter(fake_loader)
    for i, (real_data, _) in enumerate(tqdm(real_loader, mininterval=2, desc=' - Training', leave=False)):
        try:
            fake_data = next(fake_iter)
        except:
            fake_iter = iter(fake_loader)
            fake_data = next(fake_iter)

        fake_data = Variable(fake_data[0])
        real_data = Variable(real_data)

        if opt.cuda:
            fake_data = fake_data.cuda()

        data = torch.vstack((fake_data,real_data))
        target = [0 for _ in range(len(fake_data))] +\
                        [1 for _ in range(len(real_data))]
        target = Variable(torch.Tensor(target)).type(torch.long)

        if opt.cuda:
                data, target = data.cuda(), target.cuda()

        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_words

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for _, data,_ in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            h0 = Variable(torch.zeros((1, len(data), g_hidden_dim)))
            if opt.cuda:
                data = data.cuda()
                h0 = h0.cuda()
            pred = model.forward(data,h0)
            loss = criterion(pred[:-1], data.view(-1)[1:])
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.bool)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###############################################################################
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    train_iter = datasets.AG_NEWS(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>","<pad>","<sos>"], min_freq=7)
    vocab.set_default_index(vocab["<unk>"])

    UNK, PAD, SOS = 0, 1, 2

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    def collate_batch(batch, SEQ_LEN):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            processed_text = torch.cat((torch.tensor([SOS]),processed_text))
            if len(processed_text)>SEQ_LEN:
                processed_text = processed_text[:SEQ_LEN]
            else:
                pad = torch.tensor(PAD).repeat(SEQ_LEN-len(processed_text))
                processed_text = torch.cat((processed_text,pad))
            text_list.append(processed_text)

        label_list = torch.tensor(label_list, dtype=torch.int64)

        text_list = torch.stack(text_list)
        return text_list.to(device), label_list.to(device)


    train_iter, test_iter = datasets.AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    custom_collate_batch = lambda batch: collate_batch(batch,SEQ_LEN+1)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_batch)
    # test_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_batch)

    # batch = next(iter(train_loader))
    # print()
    VOCAB_SIZE = len(vocab)
    print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))
    vocab.itos = vocab.get_itos()


    # Undersampling
    positive_subset = [i for i in train_dataset if np.in1d(i[0],[3,4])]
    subset = positive_subset
    if opt.u_sample_ratio != 1:
        count = int(len(subset)/opt.u_sample_ratio)
        subset_idx = np.random.choice(range(len(subset)),count)
        subset = np.array(subset)[subset_idx]
    
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_batch)
    batch = next(iter(train_loader))
    print(' '.join([vocab.itos[i] for i in batch[1][0]]))
    ###############################################################################
    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    ###############################################################################

    print(' '.join([vocab.itos[i] for i in generator.sample(1, SEQ_LEN, SOS)[0]]))
    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters(),lr=0.01)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, train_loader, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        # loss = eval_epoch(generator, train_loader, gen_criterion)
        # print('Epoch [%d] True Loss: %f' % (epoch, loss))
        print(' '.join([vocab.itos[i] for i in generator.sample(1, SEQ_LEN, SOS)[0]]))

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=1e-4)
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(5):
        fake_samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, SOS)
        
        fake_samples = torch.Tensor(fake_samples).long()
        fake_dataset = TensorDataset(fake_samples)
        fake_loader = DataLoader(fake_dataset,batch_size=BATCH_SIZE)

        for _ in range(3):
            loss = discriminator_train_epoch(discriminator, train_loader, fake_loader, dis_criterion, dis_optimizer)
            print('Epoch [%d], loss: %f' % (epoch, loss))

    # Adversarial Training
    rollout = Rollout(generator, 0.2, SOS) # set 0 for the exact paper mehtod, But in author's official code, it's set to 0.8
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(),lr=1e-4)
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(reduction='sum')
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=1e-4)
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, SEQ_LEN, SOS)
            zeros = Variable(torch.tensor(SOS).repeat((BATCH_SIZE, 1)).type(torch.LongTensor))
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view(-1)

            # calculate the reward
            input_rewards = Variable(torch.cat([zeros, samples.data], dim = 1).contiguous())
            rewards = rollout.get_reward(input_rewards, 1, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            h0 = Variable(torch.zeros((1, BATCH_SIZE, g_hidden_dim)))
            if opt.cuda:
                rewards = rewards.cuda()
                h0 = h0.cuda()
            prob = generator.forward(inputs, h0)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            loss = eval_epoch(generator, train_loader, gen_criterion)
            print('Epoch [%d] True Loss: %f' % (total_batch, loss))
            print(' '.join([vocab.itos[i] for i in generator.sample(1, SEQ_LEN, SOS)[0]]))



        rollout.update_params()

        for _ in range(3):
            fake_samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, SOS)
            
            fake_samples = torch.Tensor(fake_samples).long()
            fake_dataset = TensorDataset(fake_samples)
            fake_loader = DataLoader(fake_dataset,batch_size=BATCH_SIZE)

            for _ in range(2):
                loss = discriminator_train_epoch(discriminator, train_loader, fake_loader, dis_criterion, dis_optimizer)
                print('Epoch [%d], Discriminator loss: %f' % (total_batch, loss))

        if total_batch % 10 == 0:
            torch.save(generator.state_dict(),'./log/gen_{}_{:04d}.t7'.format(opt.u_sample_ratio,total_batch))  
            torch.save(discriminator.state_dict(),'./log/disc_{}_{:04d}.t7'.format(opt.u_sample_ratio,total_batch))                
if __name__ == '__main__':
    main()
