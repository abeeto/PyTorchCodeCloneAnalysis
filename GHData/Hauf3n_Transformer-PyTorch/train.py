import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import argparse
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Transformer import Transformer
from Language_DataSet import Language_DataSet

device = torch.device("cuda:0")
dtype = torch.float

# embedding size of fasttext models
d_model = 300 

def train(arguments):
    
    # init arguments and hyperparameters
    
    path = arguments.path
    report = arguments.report
    
    # select language
    language_in = arguments.language_in
    language_out = arguments.language_out
    
    # training
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    
    # learning rate 
    lr = arguments.lr
    scheduler_step_size = arguments.scheduler_step_size
    scheduler_gamma = arguments.scheduler_gamma
    
    # model size
    num_heads = arguments.num_heads
    num_cells = arguments.num_cells
    cell_embedding_size = arguments.cell_embedding_size
    
    # init train
    
    # vocabulary infos
    
    vocab_in = joblib.load(path+f'vocab_{language_in}.data')
    vocab_out = joblib.load(path+f'vocab_{language_out}.data')
    
    seq_len_encoder = vocab_in["max_sentence_len"] + 1
    seq_len_decoder = vocab_out["max_sentence_len"] + 1
    vocab_size = vocab_out["vocab_size"] + 1 # + <EOS>
    
    # build model
    transformer = Transformer(num_cells,num_heads,
                          seq_len_encoder,seq_len_decoder,
                          cell_embedding_size,vocab_size).to(device)
    
    print("GPU: ", torch.cuda.get_device_name(device=None))
    print("transformer weights: ",sum(p.numel() for p in transformer.parameters() if p.requires_grad))

    # dataset
    dataset = Language_DataSet(path, language_in, language_out)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    
    # optimizer and loss objective
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, gamma=scheduler_gamma)
    
    loss_log = []
    # train
    for epoch in range(epochs):
        print(f'epoch: {epoch} | learning rate: {scheduler.get_lr()[0]}')
        #print(f'epoch: {epoch}')
        
        epoch_start = time.time()
        
        for i, batch in enumerate(dataloader):
            batch_start = time.time()
            optimizer.zero_grad()
            
            # data
            x_in, x_out, target = batch
            
            # reshape target for cross entropy loss
            target = torch.flatten(target)
            
            # run transformer
            out = transformer(x_in, x_out)
            
            #loss
            loss = cross_entropy_loss(out, target)
            loss_log.append(loss.detach().cpu().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
            optimizer.step()
            
            batch_end = time.time()
            
            if i%report == 0:
                print(f'i:{i} | loss: {loss} batch time: {batch_end - batch_start}')
        
        # decrease learning rate
        scheduler.step()
        
        epoch_end = time.time()
        print(f'epoch time: {epoch_end - epoch_start}')
    
    # save model
    torch.save(transformer,f"transformer.pt")
    
    # save loss as csv
    f = open("loss_log.csv",'w')
    f.write(f'i,loss\n')
    for i,loss in enumerate(loss_log):
        f.write(f"{i},{loss}\n")
    f.close()
    
if __name__ == "__main__":
    
    arguments = argparse.ArgumentParser()
    
    # set hyperparameters
    
    arguments.add_argument('-lr', type=float, default=1.5e-4)
    arguments.add_argument('-scheduler_step_size', type=int, default=2)
    arguments.add_argument('-scheduler_gamma', type=float, default=0.96)
    arguments.add_argument('-batch_size', type=int, default=128)
    arguments.add_argument('-path', default='')
    arguments.add_argument('-epochs', type=int, default=40)
    arguments.add_argument('-num_cells', type=int, default=6)
    arguments.add_argument('-cell_embedding_size', type=int, default=64)
    arguments.add_argument('-num_heads', type=int, default=8)
    arguments.add_argument('-language_in', default='de')
    arguments.add_argument('-language_out', default='en')
    arguments.add_argument('-report', type=int, default=250)
    
    train(arguments.parse_args())