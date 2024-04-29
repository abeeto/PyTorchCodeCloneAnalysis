# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload 2
#%%
import torchvision

base_net = torchvision.models.resnet50(pretrained=True)
print(list(base_net.children()))



#%% Test Dataset

from src.datasets import IRMAS

ds = IRMAS(is_test=True)
print(len(ds))

audio, fs = ds.get_audio(10)

#mel_spec, label = ds[0]
