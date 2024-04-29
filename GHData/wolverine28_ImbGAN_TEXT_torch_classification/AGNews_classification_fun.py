# -*- coding:utf-8 -*-

import os
import random
import math
import dill 

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset

from classifier import Classifier
from utils import get_oversampled_loader,  save_checkpoint, get_imbalanced_loader, MyDataset_origin, get_overlapped_datset, get_datset, collate_batch, MyDataset
from word_correction import tokenizer

from generator import Generator
from discriminator import Discriminator
from latent_mapping import latent_mapping

from GAN_training_fun import train_GAN_SUB

# ================== utils =================
def compute_BA(preds, labels):

    TP = torch.logical_and(preds==1,labels==1).sum()
    FP = torch.logical_and(preds==1,labels==0).sum()
    TN = torch.logical_and(preds==0,labels==0).sum()
    FN = torch.logical_and(preds==0,labels==1).sum()

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)

    # if target.sum()!=0:

    BA = (TPR+TNR)/2
    return BA

def run(IR,rep,GPU_NUM,OR,Method):

    output_file = '{:s}/rep_{:02d}_IR_{:.4f}_{:s}_AGNews.csv'.format('./output',rep,IR,Method)
    print(output_file)
    if os.path.exists(output_file):
        return 0
    
    if Method not in ['Original', 'ROS', 'GAN', 'ImbGAN']:
        return 0

    #########################################################################################
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    #########################################################################################

    # ================== Parameter Definition =================
    USE_CUDA = True
    BATCH_SIZE = 128
    SEQ_LEN = 100
    GENERATED_NUM = 10000

    emb_dim = 200
    hidden_dim = 100
    u_sample_ratio = IR

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

    use_cuda = True
    # ================== Dataset Definition =================
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    train_iter = datasets.AG_NEWS(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>","<pad>","<sos>"], min_freq=7)
    vocab.set_default_index(vocab["<unk>"])

    UNK, PAD, SOS = 0, 1, 2

    train_iter, test_iter = datasets.AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    VOCAB_SIZE = len(vocab)
    print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))
    vocab.itos = vocab.get_itos()

    # Undersampling
    negative_subset = [i for i in train_dataset if np.in1d(i[0],[1,2])]
    positive_subset = [i for i in train_dataset if np.in1d(i[0],[3,4])]
    subset = positive_subset
    if u_sample_ratio != 1:
        count = int(len(subset)/u_sample_ratio)
        subset_idx = np.random.choice(range(len(subset)),count)
        subset = np.array(subset)[subset_idx]
    train_dataset = np.concatenate((np.array(negative_subset),subset))

    ori_label = train_dataset[:,0].astype(np.int)
    bin_label = (ori_label>=3)*1
    train_dataset = MyDataset_origin(train_dataset[:,1],bin_label,ori_label)

    negative_subset = [i for i in test_dataset if np.in1d(i[0],[1,2])]
    positive_subset = [i for i in test_dataset if np.in1d(i[0],[3,4])]
    subset = positive_subset
    if u_sample_ratio != 1:
        count = int(len(subset)/u_sample_ratio)
        subset_idx = np.random.choice(range(len(subset)),count)
        subset = np.array(subset)[subset_idx]
    test_dataset = np.concatenate((np.array(negative_subset),subset))

    ori_label = test_dataset[:,0].astype(np.int)
    bin_label = (ori_label>=3)*1
    test_dataset = MyDataset_origin(test_dataset[:,1],bin_label,ori_label)

    # ================== Dataloader Definition =================
    if OR:
        train_dataset = get_overlapped_datset(train_dataset,3)
        test_dataset = get_datset(test_dataset)
    else:
        train_dataset = get_datset(train_dataset)
        test_dataset = get_datset(test_dataset)

    if Method!='ROS':
        train_loader, test_loader = get_imbalanced_loader(train_dataset,test_dataset,BATCH_SIZE,SEQ_LEN+1,vocab)
    else:
        train_loader, test_loader = get_oversampled_loader(train_dataset,test_dataset,BATCH_SIZE,SEQ_LEN,vocab)

    # batch = next(iter(train_loader))
    # print([TEXT.vocab.itos[i] for i in batch[0][0]])

    # ================== Model Definition =================
    classifier = Classifier(num_voca=VOCAB_SIZE,emb_dim=emb_dim,hidden_dim=hidden_dim,use_cuda=USE_CUDA)
    criterion = nn.BCEWithLogitsLoss()
    if USE_CUDA:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(classifier.parameters(),lr=1e-4)

    if Method=='ImbGAN' or Method=='GAN':
        generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, use_cuda)
        generator.load_state_dict(torch.load('./log/gen_10_0070.t7').state_dict())
        if use_cuda:
            generator = generator.cuda()

    if Method=='ImbGAN':
        netSubD = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
        netM = latent_mapping(g_hidden_dim)
        if use_cuda:
            netSubD = netSubD.cuda()
            netM = netM.cuda()
        optimizerSubD = optim.Adam(netSubD.parameters(),lr=1e-4)
        optimizerM = optim.Adam(netM.parameters(),lr=1e-4)

    # ================== Oversampling Set  =================
    if Method=='GAN' or Method=='ImbGAN':
        def generate_samples(model,h, batch_size, generated_num, SOS):
            samples = []
            for _ in range(int(generated_num / batch_size)):
                sample = model.sample(h, batch_size, SEQ_LEN, SOS).cpu().data.numpy().tolist()
                samples.extend(sample)
            return np.array(samples)
        h = torch.randn((1, BATCH_SIZE, g_hidden_dim))
        m_prime = torch.tensor(generate_samples(generator,h, BATCH_SIZE, GENERATED_NUM, SOS))
        m_prime_miss = torch.tensor(generate_samples(generator,h, BATCH_SIZE, GENERATED_NUM, SOS))
        train_dataset_processed = collate_batch(train_dataset, SEQ_LEN+1, vocab)
        m_miss = train_dataset_processed[1][train_dataset_processed[0]==1].cuda()
        m_miss = m_miss[:,1:]

    # print(' '.join([vocab.itos[i] for i in generator.sample(1, SEQ_LEN, SOS)[0]]))
    # ================== Training Loop =================
    N_EPOCH = 50
    beta = 0.5
    for i in range(N_EPOCH):
        classifier.train()
        train_acc, train_loss  = [], []
        all_miss = []
        for batch_no, batch in enumerate(tqdm(train_loader,total=int(len(train_loader)/BATCH_SIZE))):
            target = batch[0]
            data = batch[1]
            if USE_CUDA:
                data, target = data.cuda(), target.float().cuda()

            if (Method=='GAN' or Method=='ImbGAN') and i>25:
                ## Prepare training set
                ID = int(((target==0).sum()-(target==1).sum()).item()*1)
                if ID<=0:
                    ID=0
                ID_origin = int(ID*(1-beta))
                ID_overlap = ID-ID_origin
                t_m_prime = m_prime[np.random.choice(range(m_prime.shape[0]),ID_origin)]
                t_m_prime_miss = m_prime_miss[np.random.choice(range(m_prime_miss.shape[0]),ID_overlap)]
                syn_data = torch.cat((t_m_prime,t_m_prime_miss))

                all_data = torch.cat((data[:,1:],syn_data.cuda()))
                all_target = torch.cat((target,torch.ones(syn_data.shape[0]).cuda()))
            else:
                all_data = data[:,1:]
                all_target = target

            ## Update Classifier
            optimizer.zero_grad()
            pred = classifier(all_data)
            loss = criterion(pred.view(-1),all_target)
            loss.backward()
            optimizer.step()

            ## Evaluation
            predicted = torch.round(torch.sigmoid(pred.data)).view(-1)
            acc = compute_BA(predicted,all_target)
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            ## Update miss set
            all_miss.append(all_data[torch.logical_and(all_target==1,predicted==False)])

        if Method=='ImbGAN':
            batch_miss = torch.cat(all_miss)
            alpha = (1+np.cos((i)/N_EPOCH*np.pi))*0.5
            # l = np.random.choice(range(batch_miss.shape[0]),int(np.ceil(batch_miss.shape[0]*alpha)), replace=False)
            m_count = min(int(np.ceil(m_miss.shape[0]*alpha)),int(np.ceil(batch_miss.shape[0]*alpha)))
            l = np.random.choice(range(batch_miss.shape[0]),m_count, replace=False)
            idx = np.random.choice(range(m_miss.shape[0]),len(l), replace=False)
            m_miss[idx] = batch_miss[l]    

        train_epoch_loss = np.mean( train_loss )
        train_epoch_acc = np.mean( train_acc )

        # eval test
        classifier.eval()
        with torch.no_grad():
            test_pred, test_label = [], []
            test_loss = []
            for batch in tqdm(test_loader,total=int(len(test_loader)/BATCH_SIZE)):
                text = batch[1][:,1:]
                label = batch[0]
                if USE_CUDA:
                    text, label = text.cuda(), label.float().cuda()
                
                pred = classifier(text)
                loss = criterion(pred.view(-1),label)
                predicted = torch.round(torch.sigmoid(pred.data)).view(-1)


                test_pred.append(predicted)
                test_label.append(label)
                test_loss.append(loss.item())
            test_pred = torch.cat(test_pred)
            test_label = torch.cat(test_label)

            acc = compute_BA(test_pred,test_label)
            print('epoch:{}/{} epoch_train_loss:{:.4f},epoch_train_acc:{:.4f}'
                ' epoch_val_loss:{:.4f},epoch_val_acc:{:.4f}'.format(i+1, N_EPOCH,
                    train_epoch_loss.item(), train_epoch_acc.item(),
                    np.mean(test_loss), acc))

        ## Update Generator
        if Method=='ImbGAN':
            if i > 25:
                if i%5==4:
                    m_miss_dataset = MyDataset(m_miss,torch.ones(m_miss.shape[0]))
                    m_miss_dataloader = torch.utils.data.DataLoader(m_miss_dataset, batch_size=128,
                                                        shuffle=True, num_workers=int(0))
                    print('Training GAN model')

                    train_GAN_SUB(g_hidden_dim, generator, netM, optimizerM, netSubD, optimizerSubD,m_miss_dataloader, SOS, 10, BATCH_SIZE, SEQ_LEN, vocab, use_cuda=True)
                    # train_WGANGP(128,netG,netD,25, m_miss_dataloader,optimizerD, optimizerG, CRITIC_ITERS=5,LAMBDA=5,opt_outf=None)
                    z = torch.randn((BATCH_SIZE, g_hidden_dim*2)).cuda()
                    sub_z = netM(z)
                    m_prime_miss = generate_samples(generator, sub_z, BATCH_SIZE, GENERATED_NUM, SOS)
                    generator.load_state_dict(torch.load('./log/gen_10_0070.t7').state_dict())
            


    pred_res = torch.stack((test_pred,test_label)).T
    pred_res = pred_res.cpu().numpy().astype(np.int8)
    np.savetxt(output_file, pred_res,delimiter=',')


if __name__=="__main__":
    m = ['Original', 'ROS', 'GAN', 'ImbGAN']
    run(10,0,0,False,m[3])