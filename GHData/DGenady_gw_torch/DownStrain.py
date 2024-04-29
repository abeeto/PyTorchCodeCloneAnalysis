import os.path
import time
import urllib
import boto3
import argparse


parser = argparse.ArgumentParser(description='PyTorch Triplet network')
parser.add_argument('--start-file', type=int, default=0, metavar='N',
                    help='first file to upload to S3 storage (default: 0)')
parser.add_argument('--last-file', type=int, default=10, metavar='N',
                    help='last file to upload to S3 storage (default: 10)')
args = parser.parse_args()

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')

with open('gw_torch/download/O1_H1.txt','r') as f:
    H1_list = f.read().split('\n')
    
with open('gw_torch/download/O1_L1.txt','r') as f:
    L1_list = f.read().split('\n')
    
for i in range(args.start_file,args.last_file):
    H1_file = H1_list[i]
    H1_save = 'H1_' + H1_list[i].split('-')[-2] + '.hdf5'
    urllib.request.urlretrieve(H1_file, f'{H1_save}')
    
    L1_file = L1_list[i]
    L1_save = 'L1_' + L1_list[i].split('-')[-2] + '.hdf5'
    urllib.request.urlretrieve(L1_file, f'{L1_save}')
    
    file_present = False

    while file_present == False:
        
        if os.path.isfile(H1_save):
            s3.meta.client.upload_file(f'{H1_save}','tau-astro', f'gdevit/download/H1/{H1_save}')
            file_present = True
            break
            
        time.sleep(5)
        
    file_present = False

    while file_present == False:
        
        if os.path.isfile(L1_save):
            s3.meta.client.upload_file(f'{L1_save}','tau-astro', f'gdevit/download/L1/{L1_save}')
            file_present = True
            break
            
        time.sleep(5)
           
