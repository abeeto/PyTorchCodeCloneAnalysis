#%%
import torch.nn as nn
import torch
from torchvision.models import vgg19
from config import device

class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        self.vgg54 = vgg.features[:36].eval().to(device)
        self.mseLoss = nn.MSELoss()

        for param in self.vgg54.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        vgg_hr = self.vgg54(hr)
        vgg_sr = self.vgg54(sr)
        return self.mseLoss(vgg_hr, vgg_sr)


vgg = VggLoss().to(device)
BCELoss = nn.BCEWithLogitsLoss().to(device)
mseLoss = nn.MSELoss().to(device)
maeLoss = nn.L1Loss().to(device)

def vgg_loss(sr,hr):
    return vgg(sr,hr)

def adversarial_loss(disc_sr):
    return BCELoss(disc_sr, torch.ones_like(disc_sr))

def discriminator_loss(disc_hr, disc_sr):
    #bce_hr = BCELoss(disc_hr, torch.ones_like(disc_hr)- 0.1 * torch.rand_like(disc_hr))
    bce_hr = BCELoss(disc_hr, torch.ones_like(disc_hr))
    bce_sr = BCELoss(disc_sr, torch.zeros_like(disc_sr))
    return bce_hr + bce_sr

def discriminator_accuracy(disc_hr, disc_sr):
    size = disc_hr.size(0)
    hr_pred = disc_hr.sigmoid().round()
    sr_pred = disc_sr.sigmoid().round()
    hr_correct = hr_pred.sum()
    sr_correct = size - sr_pred.sum()
    return (hr_correct + sr_correct)/(2*size)
