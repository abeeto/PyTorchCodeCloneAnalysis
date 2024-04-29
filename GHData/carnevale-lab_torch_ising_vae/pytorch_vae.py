import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Model import Model
from copy import deepcopy

from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from pathlib import Path
from config import Config
from helpers.helpers import save_pickle, load_pickle, loss_function, save_npy, seqs_to_txt, check_act
from helpers.plotters import Plotter
from helpers.bvms import get_bvms, get_covars
from scipy.stats import pearsonr
import sys

import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_path = "configs/" + str(sys.argv[1])
conf = Config.from_json_file(Path(conf_path))

datapath = "data/"
lr = 1e-3
out_name = "E"+str(conf.EPOCHS)+"_B"+str(conf.BATCH_SIZE)+"_D"+str(conf.LAYERS)+"_N"+str(conf.NUM_NODES)+"_L"+str(conf.LATENT)+"_"+conf.ACTIVATE
Path(out_name).mkdir(parents=True, exist_ok=True)
kwargs = {'num_workers': 1, 'pin_memory': True} 

arr = np.load(datapath + 'ising.npy')
np.random.shuffle(arr)
data_array = torch.from_numpy(arr)
train, val = random_split(data_array,[25000,25000])


train_loader = DataLoader(train, batch_size=conf.BATCH_SIZE)
val_loader  = DataLoader(val,  batch_size=conf.BATCH_SIZE)
test_loader = DataLoader(data_array, batch_size=conf.BATCH_SIZE)

enc = Encoder(enc_dim=conf.LAYERS, input_dim=conf.IN_DIM, latent_dim=conf.LATENT,
activate=check_act(conf.ACTIVATE), node_count=conf.NUM_NODES)

dec = Decoder(dec_dim=conf.LAYERS, output_dim=conf.IN_DIM, latent_dim=conf.LATENT,
activate=check_act(conf.ACTIVATE), node_count=conf.NUM_NODES)

print("Device: " + str(device))

model = Model(Encoder=enc, Decoder=dec, device=device, conf=conf).to(device)

plotter = Plotter(conf)

# if len(sys.argv) == 3:
#     load_pickle(model, out_name)
#     print(model)
#     summary(model,col_names=["kernel_size", "num_params"])

# elif len(sys.argv) == 2:
print(model)
summary(model,col_names=["kernel_size", "num_params"])
optimizer = Adam(model.parameters(), lr=lr)
print("Start training VAE...")
model.train()

train_loss_arr, val_loss_arr = model.trainer_func(optimizer, train_loader, val_loader)
save_pickle(model, out_name)

plotter.loss_plot(train_loss_arr, val_loss_arr, out_name)

temp_str = ""
for i,item in enumerate(train_loss_arr):
    if i:
        temp_str = temp_str + ":"
    temp_str = temp_str + str(round(item,3))

temp_str=temp_str+","

for j,item in enumerate(val_loss_arr):
    if j:
        temp_str = temp_str + ":"
    temp_str += str(round(item,3))

with open(sys.argv[2], "a") as h:
    h.write("\n"+str(model.epochs)+","+str(model.batch_size)+","+str(model.hidden_dim)+","+str(model.n_count)+","+str(model.l_dim)+","+conf.ACTIVATE+","+temp_str)

  


mean, log_var = enc.generator(data_array, device)
print("Enc Generation Complete")
gend = dec.generator(model.l_dim, model.batch_size, device)
save_npy(out_name+"/genSeqs.npy", gend)
print("Dec Generation Complete")

seqs_to_txt(out_name)
'''
mag = plotter.init_mag(data_array)

if conf.LPLOT:
    print("Beginning Latent Plot")
    plotter.latent_plot(mean, log_var, mag, out_name)
    print("Latent Plot Complete")

if conf.TSNE:
    print("Beginning TSNE Plot")
    plotter.tsne_plot(mean, mag, out_name)
    print("TSNE Plot Complete")

if conf.HAMMING:
    print("Beginning Hamming Distance Plot")
    seqs_to_txt(out_name)
    plotter.hamming_plot(out_name)
    print("Hamming Distance Plot Complete")

if conf.COVARS:
    # if Path(out_name+"covars_Original.npy").exists():
    # if Path(out_name+"bvms_Original.npy").exists():
    biv_orig = get_bvms("Original", "orig.txt", out_name, out_name, 2, 50000)
    get_covars("Original", biv_orig, out_name)
    biv_pred = get_bvms("Predicted", "pred.txt", out_name, out_name, 2, 50000)
    get_covars("Predicted", biv_pred, out_name)

    target_covars = np.load(out_name + "/covars_Original.npy")
    target_masked = np.ma.masked_inside(target_covars, -0.01, 0.01).ravel()
    target_covars = target_covars.ravel()
    pred_covars = np.load(out_name + "/covars_Predicted.npy")
    pred_masked = np.ma.masked_inside(pred_covars, -0.01, 0.01).ravel()
    pred_covars = pred_covars.ravel()

    pearson_r, pearson_p = pearsonr(target_covars, pred_covars)
    print(pearson_r, pearson_p)

    with open("covars.txt", "a") as h:
        h.write("\n"+str(model.epochs)+","+str(model.batch_size)+","+str(model.hidden_dim)+","+str(model.n_count)+","+str(model.l_dim)+","+conf.ACTIVATE+","+str(pearson_r))

print("Finish!")
'''