import copy
import datetime
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as opt
from tqdm import tqdm

from colorama import Fore, Style
import torchinfo
import torchsummary
from torch.optim import lr_scheduler
from model.physnet_torch import PhysNet
from utils.preprocessor import preprocessing
from dataset.datasetloader import datasetloader
from utils.losses import neg_pear_loss
from utils.funcs import normalize, detrend, plot_graph

with open('params.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
    meta_params = jsonObject.get("meta_params")
#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,9"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())  # 출력결과: 2 (2, 3 두개 사용하므로)
print('Current cuda device:', torch.cuda.current_device())

'''
Setting Learning Model
'''
if __TIME__:
    start_time = time.time()
model = PhysNet()
if meta_params["pre_trained"] == 1:
    model.load_state_dict(torch.load('./checkpoints/pretrained_physnet.pth'))
else:
    print('Not using any pretrained models')

model = model.cuda()
model = nn.DataParallel(model).to(device)

if __MODEL_SUMMARY__:
    torchinfo.summary(model, (1, 3, 32, 128, 128))

if __TIME__:
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "model initialize time \t: " + Style.RESET_ALL,
          datetime.timedelta(seconds=time.time() - start_time))
'''
Generate preprocessed data hpy file 
'''
if __PREPROCESSING__:
    if __TIME__:
        start_time = time.time()

    preprocessing(save_root_path=params["save_root_path"],
                  dataset_root_path=params["data_root_path"],
                  train_ratio=params["train_ratio"])
    if __TIME__:
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "preprocessing time \t:" + Style.RESET_ALL,
              datetime.timedelta(seconds=time.time() - start_time))
'''
Load dataset before using Torch DataLoader
'''
if __TIME__:
    start_time = time.time()

dataset = datasetloader(save_root_path=params["save_root_path"],
                        option="train"
                        )

train_dataset, validation_dataset = random_split(dataset,
                                                 [int(np.floor(
                                                     len(dataset) * params["validation_ratio"])),
                                                     int(np.ceil(
                                                         len(dataset) * (1 - params["validation_ratio"])))]
                                                 )
if __TIME__:
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "load train hpy time \t: " + Style.RESET_ALL,
          datetime.timedelta(seconds=time.time() - start_time))

if __TIME__:
    start_time = time.time()
test_dataset = datasetloader(save_root_path=params["save_root_path"],
                             option="test"
                             )
if __TIME__:
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "load test hpy time \t: " + Style.RESET_ALL,
          datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if __TIME__:
    start_time = time.time()
train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"],
                          shuffle=params["train_shuffle"])
validation_loader = DataLoader(validation_dataset, batch_size=params["train_batch_size"],
                               shuffle=params["train_shuffle"])
test_loader = DataLoader(test_dataset, batch_size=params["test_batch_size"],
                         shuffle=params["test_shuffle"])
if __TIME__:
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "generate dataloader time \t: " + Style.RESET_ALL,
          datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Loss Function
'''
if __TIME__:
    start_time = time.time()
criterion = neg_pear_loss()

if __TIME__:
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "setting loss func time \t: " + Style.RESET_ALL,
          datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Optimizer
'''
if __TIME__:
    start_time = time.time()
optimizer = opt.Adam(model.parameters(), hyper_params["learning_rate"])
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
if __TIME__:
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "setting optimizer time \t: " + Style.RESET_ALL,
          datetime.timedelta(seconds=time.time() - start_time))
'''
Model Training Step
'''
min_val_loss = 100.0
min_val_loss_model = None

for epoch in range(hyper_params["epochs"]):
    if __TIME__ and epoch == 0:
        start_time = time.time()
    with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
        model.train()
        running_loss = 0.0
        i = 0
        for inputs, target in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            outputs = model(inputs)

            if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys"]:
                loss = criterion(outputs, target)
            else:
                loss_0 = criterion(outputs[:][0], target[:][0])
                loss_1 = criterion(outputs[:][1], target[:][1])
                loss_2 = criterion(outputs[:][2], target[:][2])
                loss = loss_0 + loss_2 + loss_1

            if ~torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            tepoch.set_postfix(loss=running_loss / params["train_batch_size"])
    if __TIME__ and epoch == 0:
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "1 epoch training time \t: " + Style.RESET_ALL,
              datetime.timedelta(seconds=time.time() - start_time))
        running_loss = 0.0
        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(f"Validation")
                outputs = model(inputs)
                if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys"]:
                    loss = criterion(outputs, target)
                else:
                    loss_0 = criterion(outputs[:][0], target[:][0])
                    loss_1 = criterion(outputs[:][1], target[:][1])
                    loss_2 = criterion(outputs[:][2], target[:][2])
                    loss = loss_0 + loss_2 + loss_1

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / params["train_batch_size"])
            if min_val_loss > running_loss:  # save the train model
                min_val_loss = running_loss
                checkpoint = {'Epoch': epoch,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, params["checkpoint_path"] + model_params["name"] + "/"
                           + params["dataset_name"] + "_" + str(epoch) + "_"
                           + str(min_val_loss) + '.pth')
                min_val_loss_model = copy.deepcopy(model)

    if epoch + 1 == hyper_params["epochs"] or epoch % 10 == 0:
        if __TIME__ and epoch == 0:
            start_time = time.time()
        if epoch + 1 == hyper_params["epochs"]:
            model = min_val_loss_model
        with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
            model.eval()
            inference_array = []
            target_array = []
            with torch.no_grad():
                for inputs, target in tepoch:
                    tepoch.set_description(f"test")
                    outputs = model(inputs)
                    if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys"]:
                        loss = criterion(outputs, target)
                    else:
                        loss_0 = criterion(outputs[:][0], target[:][0])
                        loss_1 = criterion(outputs[:][1], target[:][1])
                        loss_2 = criterion(outputs[:][2], target[:][2])
                        loss = loss_0 + loss_2 + loss_1

                    if ~torch.isfinite(loss):
                        continue
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / (params["train_batch_size"] / params["test_batch_size"]))
                    if model_params["name"] in ["PhysNet", "PhysNet_LSTM"]:
                        inference_array.extend(normalize(outputs.cpu().numpy()[0]))
                        target_array.extend(normalize(target.cpu().numpy()[0]))
                    else:
                        inference_array.extend(outputs[:][0].cpu().numpy())
                        target_array.extend(target[:][0].cpu().numpy())
                    if tepoch.n == 0 and __TIME__:
                        save_time = time.time()

            # postprocessing
            if model_params["name"] in ["DeepPhys"]:
                inference_array = detrend(np.cumsum(inference_array), 100)
                target_array = detrend(np.cumsum(target_array), 100)

            if __TIME__ and epoch == 0:
                print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "inference time \t: " + Style.RESET_ALL,
                      datetime.timedelta(seconds=time.time() - start_time))

            plot_graph(0, 300, target_array, inference_array)
