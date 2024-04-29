import numpy as np
from sklearn.manifold import TSNE
import boto3
import torch
import argparse
from tools import *
import time

parser = argparse.ArgumentParser(description='tsne of data')

parser.add_argument('--file-path', default='Embed', type=str,
                    help='File to be embedded by t-SNE')
parser.add_argument('--label-path', default='Embed', type=str,
                    help='File to be embedded by t-SNE')
parser.add_argument('--save-name', default='Embed', type=str,
                    help='file save name')
parser.add_argument('--perplexity', type=int, default=100, metavar='N',
                    help='t-sne perplexity parameter')
parser.add_argument('--num-of-samples', type=int, default=10000, metavar='N',
                    help='how many samples to train on')

args = parser.parse_args()

start_time = time.perf_counter()

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')
file_path = args.file_path
file_name = args.save_name
label_path = args.label_path

data = torch.load(load_to_bytes(s3,f's3://tau-astro/gdevit/model/{file_path}.pt')).numpy()
labels = torch.load(load_to_bytes(s3,f's3://tau-astro/gdevit/model/{label_path}.pt')).numpy()

samples = args.num_of_samples
raw_data = np.concatenate((data[0,0,:samples,:],data[0,1,:samples,:]))
signal_data = np.concatenate((data[1,0,:samples,:],data[1,1,:samples,:]))
combined_data = np.concatenate((raw_data,signal_data))
embedded = TSNE(n_components=2, learning_rate='auto', init='random',perplexity=args.perplexity).fit_transform(combined_data)

group = np.concatenate((labels[0,0,:samples],labels[0,1,:samples],labels[1,0,:samples],labels[1,1,:samples]))
                                
np.save(f'{file_name}_embedded.npy', embedded, allow_pickle=True)
np.save(f'{file_name}_group.npy', group, allow_pickle=True)
print('Done in {} hours'.format((time.perf_counter()-start_time)/3600))
                                
