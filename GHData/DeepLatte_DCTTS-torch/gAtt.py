from params import param
import numpy as np

import torch
import matplotlib.pyplot as plt


def guideAttentionNT(g, N, T):
    Wnt = np.zeros((param.max_N, param.max_T))
    for n in range(param.max_N):
        for t in range(param.max_T):
            if t <= T:
                Wnt[n, t] = 1.0 - np.exp(-0.5 * np.power((n/N - t/T), 2) / np.power(g,2))
            else:
                Wnt[n, t] = 1.0 - np.exp(-0.5 * np.power((N-1)/N - n/N, 2) / np.power(g/2, 2)) # forcing more at end step
    
    return torch.Tensor(Wnt)


def gAttlossNT(A, Wnt, DEVICE):
    '''
    input 
        A : attention matrix (B, N, T/r)
        Wnt : guide weight (B, N, T/r)
    --------
    return
        Attention loss
    '''
    Wnt = Wnt.to(DEVICE)
    A = A.to(DEVICE)

    return torch.mean(A * Wnt)