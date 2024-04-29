import torch
import torch.nn as nn
import torch.nn.functional as F
import boto3
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
from torchvision import models
import time
from resnetNoBN import resnet50NoBN, Net
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
parser.add_argument('--num-files', default=50, type=int, metavar='N',
                    help='Number of files to embed')
parser.add_argument('--last-layer', default=32, type=int, metavar='N',
                    help='dimension of last layer')


args = parser.parse_args()


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
    
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
num_of_files = args.num_files

# first index seprates between no signal and injected signal,
# second index seperates betwenn H and L, where 0 is H and 1 is L.
embeddings = torch.empty((2,2,num_of_files*samples_in_file ,2))
latent_space =  torch.empty((2,2,num_of_files*samples_in_file ,args.last_layer))
all_labels = torch.empty((2,2,num_of_files*samples_in_file))

noise = np.load(load_to_bytes(s3,'s3://tau-astro/gdevit/model/embedded/signals.npy'), allow_pickle=True)

for file in range(0,num_of_files):
    
    data = np.load(load_to_bytes(s3,f's3://tau-astro/gdevit/{data_path}{file}.npy'), 
                   allow_pickle=True)
    data = data_to_torch(data/255)
    
    moddata = np.load(load_to_bytes(s3,f's3://tau-astro/gdevit/{data_path}{file}.npy'), 
                   allow_pickle=True)
    moddata = add_signal_class(moddata/255,noise)
    moddata= data_to_torch(moddata)
    
    
    for batch in range(samples_in_file//(batch_size)):
      
        first_ind = file*samples_in_file + batch*batch_size
        last_ind = file*samples_in_file + (batch+1)*batch_size
        
        for j,detector in enumerate(['H','L']):
            
            imgs, labels = EmbClassBatch(data, batch, batch_size,detector)
            imgs = imgs.to(device)
            results = my_model(imgs)
            all_labels[0,j, first_ind:last_ind] = labels
            embeddings[0,j, first_ind:last_ind, :] = results.detach().cpu()
            latent_space[0,j, first_ind:last_ind, :] = activation['dimRed'].detach().cpu()
            del imgs, labels, results

            imgs, labels = EmbClassBatch(moddata, batch, batch_size,detector)
            imgs = imgs.to(device)
            results = my_model(imgs)
            all_labels[1,j, first_ind:last_ind] = labels
            embeddings[1,j, first_ind:last_ind, :] = results.detach().cpu()
            latent_space[1,j, first_ind:last_ind, :] = activation['dimRed'].detach().cpu()
            del imgs, labels, results
        
torch.save(latent_space,f'{save_path}_latent.pt')
torch.save(embeddings,f'{save_path}_class.pt')
torch.save(all_labels,f'{save_path}_label.pt')
print('Done')    
