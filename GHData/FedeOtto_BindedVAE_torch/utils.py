"""
@author: Federico Ottomano
"""

import torch 
import torch.nn as nn


def loss_maskVAE(lambdaM, gamma, mask, recon_mask, mu, logvar):
    
    m_loss = torch.mean(nn.BCELoss(reduction='none')(recon_mask, mask))
    KLD = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar))
    neg_dice = negative_dice(mask, recon_mask)
    
    return lambdaM * m_loss + torch.mean(KLD) + gamma * neg_dice

def loss_ratioVAE(lambdaR, ratios, recon_mask, recon_ratio, mu, logvar):
    
    r_loss = torch.mean(nn.BCELoss(reduction='none')(recon_mask*recon_ratio, ratios))
    
    KLD = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar))
    
    return lambdaR * r_loss + torch.mean(KLD)

def negative_dice(a, b):
    
    return -2 * torch.sum(a * b) / torch.sum(a+b)