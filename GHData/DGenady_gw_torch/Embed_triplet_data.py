import torch
import torch.nn as nn
import torch.nn.functional as F
import boto3
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
from torchvision import models
import time
from resnetNoBN import resnet50NoBN, Net_trip
from tools import *
import argparse


# Class Embedding settings
parser = argparse.ArgumentParser(description='Embedding Class data')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--model-path', type=str, default='modelpath',
                    help='Select model for embedding by sepcifing its location')
parser.add_argument('--data-path', default='triplet_data/file', type=str,
                    help='folder on s3 containing data')
parser.add_argument('--start-file', default=0, type=int, metavar='N',
                    help='the index of the first file')
parser.add_argument('--end-file', default=10, type=int, metavar='N',
                    help='Index of the last file to be embedded')

args = parser.parse_args()

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')

model_name = args.model_path
data_path = args.data_path
save_path = model_name.split('/')[1]

my_model = Net(args.last_layer)
my_model.load_state_dict(torch.load(load_to_bytes(s3,f's3://tau-astro/gdevit/model/{model_name}.pt')))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

my_model.to(device)
my_model.eval()
my_model.dimRed.register_forward_hook(get_activation('dimRed'))

batch_size = args.batch_size
samples_in_file = 1000
startFile, endFild = args.start_file, args.end_file
num_of_files = endFile - startFile

embeddings = torch.empty((2,2*num_of_files*samples_in_file ,2))

noise = np.load(load_to_bytes(s3,'s3://tau-astro/gdevit/model/embedded/signals.npy'), allow_pickle=True)

for i in range(startFile, endFile):
    print(f'working on file {i}')
    
    start = (i-startFile)*SampPerFile
    data = np.load(load_to_bytes(s3,f's3://tau-astro/gdevit/{data_path}{i}.npy'), 
                   allow_pickle=True)
    data = data_to_torch(data/255)
    moddata = np.load(load_to_bytes(s3,f's3://tau-astro/gdevit/{data_path}{i}.npy'), 
                   allow_pickle=True)
    moddata = add_signal_class(moddata/255, signal)
    moddata = data_to_torch(moddata)
    
    for batch in range(SampPerFile//batchsize):
        for k in [0,1,2]:
            
            img = getBatchData(data, batch, batchsize, k) 
            img = img.to(device)
            emb = model(img)
            Embeddings[0,start+batch*batchsize:start+(batch+1)*batchsize,k,:] = emb.detach().cpu()
            del img, emb
            
            img = getBatchData(moddata, batch, batchsize, k) 
            img = img.to(device)
            emb = model(img)
            Embeddings[1,start+batch*batchsize:start+(batch+1)*batchsize,k,:] = emb.detach().cpu()
            del img, emb
        
torch.save(latent_space,f'{save_path}_latent.pt')
torch.save(embeddings,f'{save_path}_class.pt')
torch.save(all_labels,f'{save_path}_label.pt')
print('Done')    
