import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import os
import sys
sys.path.append('../')
from specAugment import specAug

import numpy as np
from params import param
from load_audio import loadTrnscpt
import matplotlib.pyplot as plt
from gAtt import guideAttentionNT

class speechDataset(Dataset):
    def __init__(self, trNet, phase):
        self.mels, self.mags = [], []

        transcript = os.path.join(param.filePath, param.transcriptName)
        _, self.scripts, self.lenList = loadTrnscpt(transcript)
        
        self.len = len(self.scripts)

        if phase is 0:
            # train
            self.start = 0
            self.end = self.len//10 * 6
        elif phase is 1:
            # validation
            self.start = self.len//10 * 6
            self.end = self.len//10 * 9
        elif phase is 2:
            # test
            self.start = self.len//10 * 9
            self.end = self.len

        self.scripts = [torch.LongTensor(self.scripts[idx]) for idx in range(self.start, self.end)]
        self.mels = [torch.tensor(np.load(os.path.join(param.filePath, "mel_dir", "mel_{}.npy".format(idx))))
                      for idx in range(self.start, self.end)]
        self.mags = [torch.tensor(np.load(os.path.join(param.filePath, "mag_dir", "mag_{}.npy".format(idx))))
                      for idx in range(self.start, self.end)]
        self.gMat = [torch.tensor(np.load(os.path.join(param.filePath, "gMat_dir", "gMat_{}.npy".format(idx))))
                      for idx in range(self.start, self.end)]

        self.trNet = trNet
        self.len = len(self.scripts)
        
    def __getitem__(self, index):
        text = self.scripts[index]
        mel = self.mels[index]
        mag = self.mags[index]
        gMat =self.gMat[index]

        if self.trNet is 't2m':
            return (text, mel, mag, gMat)
        else:
            return (text, mel, mag)
    def __len__(self):
        return self.len

def collate_fn(data):
    '''
    Deal with mini-batch which has size B
    input:
        data : [(texts, mel, mags), ( ), ( ) ... ]
                len : B
    
    return:
        texts : torch tensor of shape (B, padded_length)
        mels : torch tensor of shape(B, max_mel_len, n_mels)
        mags : torch tensor of shape(B, max_mag_len, n_mags)
        gMat : torch tensor of shape(B, max_text_len, max_mel_len)
    '''
    data.sort(key=lambda x: len(x[0]), reverse=True) # sort by length of text.
    if len(data[0]) == 3:
        texts, mels, mags = zip(*data) #upnacking operator *
    else:
        texts, mels, mags, gMat = zip(*data)

    textLen = [len(x) for x in texts]
    melLen = [len(x) for x in mels]
    magLen = [len(x) for x in mags]

    text_pads = torch.zeros(len(textLen), max(textLen), dtype=torch.long)
    mel_pads = torch.zeros(len(melLen), max(melLen), mels[0].shape[-1])
    mag_pads = torch.zeros(len(magLen), max(magLen), mags[0].shape[-1])

    for idx in range(len(textLen)):
        text_pads[idx, :textLen[idx]] = texts[idx]
        if param.specAugON:
            mel_pads[idx, :melLen[idx]] = specAug.applyAugment(mels[idx], param.spaugDomain, param.spmaskNumb)
        else:
            mel_pads[idx, :melLen[idx]] = mels[idx]
        mag_pads[idx, :magLen[idx]] = mags[idx]

    textLen_tensor = torch.LongTensor(textLen)
    if len(data[0]) == 3:
        return text_pads, mel_pads, mag_pads, textLen_tensor
    else:
        gMat_pads = torch.zeros(len(textLen), max(textLen), max(melLen))
        for idx in range(len(textLen)):
            gMat_pads[idx] = gMat[idx][:max(textLen), :max(melLen)]
        return text_pads, mel_pads, mag_pads, gMat_pads, textLen_tensor

def att2img(A):
    '''
    input: 
        A : Attention Matrix for one sentence. (1, N, T/r)
    '''
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()
    A = A[np.newaxis, :]
    T = A.shape[-1]
    for idx in range(T):
        maxA, minA = max(A[0, :, idx]), min(A[0, :, idx])
        A[0, :, idx] = (A[0, :, idx] - minA) / (maxA + 1e-5)
    return A

def plotAtt(A, text, step, modelPath):
    '''
    input : 
        A: attention matrix (1, N, T/r)
    '''

    idx2char = {idx: char for idx, char in enumerate(param.vocab)}
    
    text = text.cpu().numpy()
    text = [ idx2char[idx] for idx in text]
    textStr = ''.join(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(A[0])
    plt.title('Attention Matrix at {}\n {}'.format(step, textStr), fontsize=10)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Text', fontsize=20)
    plt.clim(0.0, 1.0)
    plt.colorbar()
    plt.yticks(np.arange(len(text)), text)
    
    plt.savefig(os.path.join(modelPath, 'Att_{}.png'.format(step)))
    plt.close()

def plotMel(mel, step, modelPath):
    '''
    input: 
        mel : mel spectrogram
              (T/r, n_mels)
    '''
    mel = mel.transpose(0,1)
    mel = mel.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(mel, origin='lower')
    plt.title('Mel spectrogram at {}'.format(step), fontsize=10)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Mel Frequency', fontsize=20)
    plt.clim(0.0, 1.0)
    plt.colorbar()
    
    plt.savefig(os.path.join(modelPath, 'Mel_{}.png'.format(step)))
    plt.close()