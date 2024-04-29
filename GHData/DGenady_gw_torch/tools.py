"""
common functions used in my scripts

"""

from urllib.parse import urlparse
from io import BytesIO
import numpy as np
import torch
from scipy.signal import chirp
import numpy as np
import boto3
import matplotlib.pyplot as plt


def loadFile(path, num, s3obj):
    """ loads files from s3 server """
    data = np.load(load_to_bytes(s3obj,f's3://tau-astro/gdevit/{path}{num}.npy'), 
                       allow_pickle=True)
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data/255

def load_to_bytes(s3,s3_uri:str):
    parsed_s3 = urlparse(s3_uri)
    f = BytesIO()
    s3.meta.client.download_fileobj(parsed_s3.netloc, 
                                    parsed_s3.path[1:], f)
    f.seek(0)
    return f


def getBatchDataClass(data,batchNum,batchsize):
    """ returns a batch of the data to be trained """
    h1 = data[batchNum*batchsize:(batchNum+1)*batchsize,0,:,:,:]
    l1 = data[batchNum*batchsize:(batchNum+1)*batchsize,2,:,:,:]
    
    h1l = torch.zeros(batchsize)
    l1l = torch.zeros(batchsize) + 1
    
    labels = torch.cat((h1l,l1l))
    img = torch.cat((h1,l1))
    
    perm = torch.randperm(img.size(0))
    
    img = img[perm,:,:,:]
    labels = labels[perm]
    return img, labels.type(torch.LongTensor)

def EmbClassBatch(data,batchNum,batchsize,k):
    """ returns a batch of the data to be trained """
    if k == 'H':
        img  = data[batchNum*batchsize:(batchNum+1)*batchsize,0,:,:,:]
        labels = torch.zeros(batchsize)
    elif k == 'L':
        img  = data[batchNum*batchsize:(batchNum+1)*batchsize,2,:,:,:]
        labels = torch.zeros(batchsize) + 1
    return img, labels.type(torch.LongTensor)



def getBatchData(data,batchNum,batchsize,k):
    imgs = data[batchNum*batchsize:(batchNum+1)*batchsize,k,:,:,:]
    return imgs

def loadFileMod(path, num, s3obj):
    """ loads files from s3 server and adds signal to the data"""
    data = np.load(load_to_bytes(s3obj,f's3://tau-astro/gdevit/{path}{num}.npy'), 
                       allow_pickle=True)
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data/255

def data_to_torch(data):
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data

def add_signal_class(original,noise):
    """
        Provided with data of shape [sample,label,height,width,color] retuns data with additonal 
        random chrip like signals to all 3 different labes.
    """
    indicies = np.arange(1000)
    np.random.shuffle(indicies)
    samples = original.shape[0]
    for sample in range(samples):
        for channel in [0,1,2]:
            index = indicies[sample]
            original[sample,channel,:,:,:] = np.maximum(original[sample,channel,:,:,:],noise[index,:,:,:])
    return original

def batch_std(samples,label,detector):
    indices = torch.where(label == detector)[0]
    return torch.mean(torch.std(samples[indices],axis=0))
