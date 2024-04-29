# %% import dependencies
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import pickle

import torch.nn as nn
import torch

import torchio as tio

from utils import train_one_epoch, valid
from dataset import Datasets3D, read_param
from model import UNet

# %% some global variables 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"model is trained on device: {device}")

# %% load dataset
dataset_path = os.path.join(os.path.dirname(__file__), "ixi_tiny")

# fetch dataset from dataset path
ixi_dataset = Datasets3D(dataset_path)
training_set = ixi_dataset.train_set()
validation_set = ixi_dataset.val_set()

# construct patch sets from dataset
patch_size = 32
samples_per_volume = 5
max_queue_length = 300
sampler = tio.data.UniformSampler(patch_size) # because ixi image size is relatively small, here we use UniformSampler
num_workers = 8

patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=True,
    shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=False,
    shuffle_patches=False,
)

# construct dataloader from patch sets
batch_size = 8
training_batch_size = batch_size
validation_batch_size = 2 * training_batch_size # validation batch size can be bigger

training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

validation_loader_patches = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)

# some other essential variables for model init or for prediction 
landmarks = ixi_dataset.get_landmarks()  # need to use the same landmarks to pre-process prediction data
num_classes = read_param(os.path.join(dataset_path, "param.json"))["num_classes"]  # need num_classes to build model
saved_params = {
    'landmarks': landmarks,
    'num_classes': num_classes
}

# %% build model
model = UNet(num_classes)
if torch.cuda.is_available():
    model = nn.DataParallel(model)
model.to(device)

optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=5, threshold=1e-3)

# %% train
epoch = 2
for epoch_ in range(epoch):
    train_loss = train_one_epoch(training_loader_patches, model, device, optimizer_ft)
    val_loss, accuracy = valid(validation_loader_patches, model, device)
    lr_scheduler.step(val_loss)

    print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch_ + 1, train_loss, val_loss, accuracy))

# %% save model params and pre_process params
if torch.cuda.is_available():
    state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()

model_param_name = f"./Unet3d_IXITiny.pth"
torch.save(state_dict, model_param_name)

landmarks_name = "./landmarks.pkl"
with open(landmarks_name, 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)