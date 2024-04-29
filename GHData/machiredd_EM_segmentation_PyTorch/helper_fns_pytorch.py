from dataset_in import EM_dataset, RandomGenerator
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from losses import DiceLoss
from simple_unet import UNet
import cv2
import logging
import sys
from utils import save_model, save_plots, SaveBestModel
from os import listdir
from scipy import signal
from os.path import join
import copy
import os
import csv
import torch.nn.functional as F

def validate(testloader, model, criterion,device):
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            counter += 1
            # forward pass
            inputs = inputs.to(device)
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, targets)
            valid_running_loss += loss.item()

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    return epoch_loss


def run_one_step(train_dl, model, optimizer, criterion, iter_no, device):
    counter = 0
    train_running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_dl):
        counter = counter + 1
        inputs = inputs.to(device)
        targets = targets.to(device)
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # loss = criterion(yhat['out'], targets)
        train_running_loss = train_running_loss + loss.item()
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        iter_no = iter_no + 1
        logging.info('iteration %d : loss : %f' % (iter_no, loss.item()))
    epoch_loss = train_running_loss / counter
    return epoch_loss,iter_no

# train the model
def train_model_new(train_dl, model, num_epochs, steps_per_epoch,dest_dir, device):
    # define the optimization
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    save_best_model = SaveBestModel()
    logging.basicConfig(filename=dest_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch. ".format(len(train_dl)))

    iter_no = 0
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        print('epoch', epoch)
        # enumerate mini batches
        per_epoch_train_loss = 0
        for steps in range(steps_per_epoch):
            train_epoch_loss,iter_no = run_one_step(train_dl, model, optimizer, criterion, iter_no, device)
            per_epoch_train_loss = per_epoch_train_loss + train_epoch_loss
        val_epoch_loss = validate(train_dl, model, criterion,device)
        avg_epoch_train_loss = per_epoch_train_loss/steps_per_epoch
        train_loss.append(avg_epoch_train_loss)
        val_loss.append(val_epoch_loss)
        logging.info('epoch %d : train_loss : %f val_loss : %f' % (epoch, avg_epoch_train_loss, val_epoch_loss))
        save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)
    # save the trained model weights for a final time
    save_model(num_epochs, model, optimizer, criterion)
    # save the loss and accuracy plots
    save_plots(train_loss, val_loss)
    print('TRAINING COMPLETE')


# train the model
def train_model_new_1(train_dl, model, num_epochs,dest_dir):
    # define the optimization
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    save_best_model = SaveBestModel()
    logging.basicConfig(filename=dest_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch. ".format(len(train_dl)))

    iter_no = 0
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        print('epoch', epoch)
        # enumerate mini batches
        train_epoch_loss = run_one_step(train_dl, model, optimizer, criterion, iter_no)
        val_epoch_loss = validate(train_dl, model, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        logging.info('epoch %d : loss : %f' % (epoch, train_epoch_loss))
        save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)
    # save the trained model weights for a final time
    save_model(num_epochs, model, optimizer, criterion)
    # save the loss and accuracy plots
    save_plots(train_loss, val_loss)
    print('TRAINING COMPLETE')


# image = cv2.imread('/Users/archana/Downloads/CycleGAN-tensorflow-master/datasets/em/testA/1.png')
# evaluate the model
def evaluate_model(test_dl, model,dest_dir,device):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        inputs = inputs.to(device)
        yhat = model(inputs)
        # retrieve numpy array
        # yhat = yhat.detach().numpy()
        # a = yhat['out']
        # yhat = F.softmax(a, dim=1)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        inputs = inputs.numpy()

        print(inputs.shape)
        print(yhat.shape)
        print(actual.shape)
        a = inputs[0, 0, :, :]
        b = actual[0, 0, :, :]
        c = yhat[0, 0, :, :]
        cv2.imwrite(dest_dir + str(i) + '_image.png', np.float32(a*255))
        cv2.imwrite(dest_dir + str(i) + '_gt.png', np.float32(b*255))
        cv2.imwrite(dest_dir + str(i) + '_pred.png', np.float32(c*255))


