import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F
# import torchvision.transforms as transforms

class param():
    filePath = '../speech_data/LJSpeech-1.1'
    transcriptName = 'metadata.csv'
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence
    vocabSize = len(vocab)

    preemphasis = 0.97
    n_fft = 2048
    n_mags = 1 + n_fft//2
    n_mels = 80
    sr = 22050 # sampling rate
    frameLen = 0.05 # length of frame
    frameStride = 0.0125 # length of stride
    winSize = int(sr*frameLen)
    hopSize = int(sr*frameStride)
    max_db = 100
    ref_db = 20
    max_N = 200 # Maximum number of characters.
    max_T = 210 # Maximum length of mel frames.
    power = 1.5 # griffin lim

    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = 512 # == hidden units of SSRN
    r = 4 # reduction factor
    B = 16 # Batch Size
    maxStep = 200000 # Epochs
    gl_iter = 50 # griffin lim iteration
    actFDic = {"ReLU" : torch.relu, "sigmoid" : torch.sigmoid}

    lr = 0.0001 # learning rate
    dr = 0.05 # dropout
    adam_beta = (0.5, 0.9)
    adam_eps = 1e-6

    #specAug
    specAugON = False
    spaugDomain = 0
    spmaskNumb = 1