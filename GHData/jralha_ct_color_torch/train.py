#%% Imports
import os
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import ct_color_torch.utils as utils
from ct_color_torch.transformer_net import TransformerNet
from ct_color_torch.vgg import Vgg16

from tqdm.auto import tqdm

#%%Vars
cuda = 0
seed = 42
image_size = 256
batch_size = 4
style_image_folder = 'ct_color_torch\\data\\style\\Libra_2-ANP-2A-RJS\\'
lr = 1e-3
dataset = 'ct_color_torch\\data\\dataset\\'
style_size = 512
epochs = 100
content_weight = 1e5
style_weight = 1e10
log_interval = 250
checkpoint_model_dir = None
checkpoint_interval = 100
save_model_dir = 'ct_color_torch\\output\\'


device = torch.device("cuda" if cuda==1 else "cpu")

np.random.seed(seed)
torch.manual_seed(seed)

 #%%Define transformations and load/transform data
transform = transforms.Compose([
    transforms.Resize(image_size), # the shorter side is resize to match image_size
    transforms.CenterCrop(image_size),
    transforms.ToTensor(), # to tensor [0,1]
    transforms.Lambda(lambda x: x.mul(255)) # convert back to [0, 255]
])
train_dataset = datasets.ImageFolder(dataset, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # to provide a batch loader

    
style_image = [f for f in os.listdir(style_image_folder)]
style_num = len(style_image)
print(str(style_num)+" style images.")

#%%Define transformer network and vgg
transformer = TransformerNet(style_num=style_num).to(device)
optimizer = Adam(transformer.parameters(), lr)
mse_loss = torch.nn.MSELoss()

vgg = Vgg16(requires_grad=False).to(device)
style_transform = transforms.Compose([
    transforms.Resize(style_size), 
    transforms.CenterCrop(style_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

#%%Loading style images
style_batch = []

for i in tqdm(range(style_num),position=0):
    style = utils.load_image(style_image_folder + style_image[i], size=style_size)
    style = style_transform(style)
    style_batch.append(style)

style = torch.stack(style_batch).to(device)

features_style = vgg(utils.normalize_batch(style))
gram_style = [utils.gram_matrix(y) for y in features_style]


#%%Training loop
for e in tqdm(range(epochs),position=0):
    transformer.train()
    agg_content_loss = 0.
    agg_style_loss = 0.
    count = 0
    for batch_id, (x, _) in enumerate(train_loader):
        n_batch = len(x)
            
        if n_batch < batch_size:
            break # skip to next epoch when no enough images left in the last batch of current epoch

        count += n_batch
        optimizer.zero_grad() # initialize with zero gradients

        batch_style_id = [i % style_num for i in range(count-n_batch, count)]
        y = transformer(x.to(device), style_id = batch_style_id)

        y = utils.normalize_batch(y)
        x = utils.normalize_batch(x)

        features_y = vgg(y.to(device))
        features_x = vgg(x.to(device))
        content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[batch_style_id, :, :])
        style_loss *= style_weight

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

        agg_content_loss += content_loss.item()
        agg_style_loss += style_loss.item()

        if (batch_id + 1) % log_interval == 0:
            mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss) / (batch_id + 1)
            )
            print(mesg)

        if checkpoint_model_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
            transformer.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
            ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
            torch.save(transformer.state_dict(), ckpt_model_path)
            transformer.to(device).train()


#%% save model
transformer.eval().cpu()
save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '') + "_" + str(int(
    content_weight)) + "_" + str(int(style_weight)) + ".model"
save_model_path = os.path.join(save_model_dir, save_model_filename)
torch.save(transformer.state_dict(), save_model_path)

print("\nDone, trained model saved at", save_model_path)

# %%
 