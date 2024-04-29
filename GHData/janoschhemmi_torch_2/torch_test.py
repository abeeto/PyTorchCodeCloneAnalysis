
import matplotlib as mpl
# mpl.use(‘TkAgg’)
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
#import unfoldNd
from inspect import getmembers, isfunction

import tsai
from tsai.all import *

import fastai
# help(fastai2)
from fastai.vision import *
from fastai.text import *
from fastai.metrics import *
from fastai.learner import *
from fastai.basics import *

import fastcore
from fastcore.all import *
# random data sample
#x = torch.range(1,20)
# unfold dimension to make our rolling window
# ie. window size of 6, step size of 1
# x.unfold(0, 6, 1)


## load tree species time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\02_df_x_10.csv', 'r') as f:
    X = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1).reshape((799,5,21))
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\02_df_Y_10.csv', 'r') as f:
    y = np.genfromtxt(f, delimiter=';',dtype=np.str, skip_header=1)

# X = torch.tensor(X)
## unfold exp

# module hyperparameters
#kernel_size = 6
#dilation = 0
#padding = 0
#stride = 0
#indices = X.shape[1]

#X = X[1,:,:]
#lib_module(X)
# both modules accept the same arguments and perform the same operation
#torch_module = torch.nn.Unfold(
#    kernel_size, dilation=dilation, padding=padding, stride=stride
#)
#lib_module = unfoldNd.UnfoldNd(
#    kernel_size, dilation=dilation, padding=padding, stride=stride
#)
#X = X [1,:,:]
#X = torch.tensor(X)
#X
#X.unfold(2, 7, 1)


## Model Test
X.shape, y.shape

splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
batch_tfms = TSStandardize(by_sample=True)

dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[50], batch_tfms=[TSStandardize()], num_workers=0)
# dls.show_batch(sharey=True,  show_title= True, ncols = 3, nrows= 5)


class FCN(Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0], dilation = 2)
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        return self.fc(x)

model_test = FCN(dls.vars, dls.c,layers=[128, 256, 128], kss=[7, 5, 3])
model_test
learner = Learner(dls, model_test, metrics=accuracy)
#learn.lr_find()


## train
#learn.fit_one_cycle(25, lr_max=1e-4)
#learner.export("P:/workspace/jan/fire_detection/dl/models_store/01_test/01_fcn.pth")

## load safed model
learner = load_learner("P:/workspace/jan/fire_detection/dl/models_store/01_test/01_fcn.pth")

## one batch
x_one,y_one = dls.one_batch()


## classes into one hot encoding
hot = torch.nn.functional.one_hot(y_one, num_classes=- 1)

## get predictions
preds,_ = learner.get_preds(dl=[(x_one,y_one)])

## get predictions only at one hot
cor_prob = torch.sum(preds * hot, dim = 1)

## stack it with prdictions
tuple = cor_prob,y_one
torch.stack(tuple,-1)

##
#trained_model.recorder.plot_metrics()
#trained_model.show_results()
#trained_model.confusion_matrix()

#interp = ClassificationInterpretation.from_learner(trained_model)
#interp.most_confused()
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


preds,_ = learner.get_preds(dl=[(x_one,y_one)])
preds[3]
learner.recorder.plot_metrics()
learner.show_results()

learner.validate(ds_idx=1, dl )
help(learner)
u = learn.x


unfold = nn.Unfold(kernel_size=(2, 3))
unfold
input = torch.randn( 10, 5, 21)

output = unfold(input)
output


#>>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
#>>> # 4 blocks (2x3 kernels) in total in the 3x4 input
#>>> output.size()
#torch.Size([2, 30, 4])

#>>> # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
#>>> inp = torch.randn(1, 3, 10, 12)
#>>> w = torch.randn(2, 3, 4, 5)
#>>> inp_unf = torch.nn.functional.unfold(inp, (4, 5))
#>>> out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
#>>> out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
#>>> # or equivalently (and avoiding a copy),
#>>> # out = out_unf.view(1, 2, 7, 8)
#>>> (torch.nn.functional.conv2d(inp, w) - out).abs().max()
#tensor(1.9073e-06)


## SPLIT WINDOW APPROACH

## loop over samples

##  make time splits for each sample

## x:: 2d array

## splits = [x[::;i:i+window_size] for i in range(0,x.size(0)-window_size+1,stride)]


## model on splits
## preds = [model.predict(i) for i in img_arr]

## safe model outputs per sampe