import argparse
from logging import BufferingFormatter
import os, pdb, sys, glob, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random
import math
import torch
import torch.nn as nn
import torchvision.models as models 
import importlib
importlib.reload(models)
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR

# import custom dataset classes
from datasets import XRaysTestDataset, XRaysTrainDataset, GANLoader, ChexpertTrainDataset, ChexpertTestDataset, CIFAR10TestDataset, CIFAR10TrainDataset


# import neccesary libraries for defining the optimizers
import torch.optim as optim

from trainer import fit_CIFAR
import config

import warnings

warnings.filterwarnings(action='ignore')

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

class weighted_loss():

    def __init__(self, pos_weights, neg_weights):
        self.pos_weights = pos_weights
        self.neg_weights = neg_weights
        print("PNB")

    def __call__(self, y_pred, y_true, epsilon=1e-7):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        sigmoid = nn.Sigmoid()
        
        for i in range(len(self.pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos =  -1 * torch.mean(self.pos_weights[i] * y_true[:, i] * torch.log(sigmoid(y_pred[:, i]) + epsilon))
            loss_neg =  -1 * torch.mean(self.neg_weights[i] * (1 - y_true[:, i]) * torch.log(1 -sigmoid( y_pred[:, i]) + epsilon))
            loss += loss_pos + loss_neg
        return loss

class Effective_PNB_loss():

    def __init__(self, pos_preq, neg_freq):
        self.beta = 0.9999999
        self.alpha = 10
        self.raw_pos_weights = self.get_inverse_effective_number(self.beta, pos_preq)
        self.neg_weights = self.get_inverse_effective_number(self.beta, neg_freq)
        
        #temp
        self.total = self.raw_pos_weights + self.neg_weights
        self.pos_weights = self.raw_pos_weights / self.total
        self.neg_weights = self.neg_weights / self.total

        print(self.raw_pos_weights)
        print(self.pos_weights)
        print(self.neg_weights)

        # self.pos_weights = [0.52583212, 0.50759632, 0.50000371, 0.60382213, 0.50036746, 0.50358188, 0.52182739, 0.50030429, 0.52754073, 0.6171878 ]
        # self.neg_weights = [0.47416788, 0.49240368, 0.49999629, 0.39617787, 0.49963254, 0.49641812, 0.47817261, 0.49969571, 0.47245927, 0.3828122 ]

    def get_inverse_effective_number(self, beta, freq): # beta is same for all classes
        sons = freq / self.alpha # scaling factor
        for i in range(len(freq)):
            sons[i] = math.pow(beta,freq[i])
        sons = np.array(sons)
        En = (1 - sons) / (1 - beta)
        return (1 / En) # the form of vector

    def __call__(self, y_pred, y_true, epsilon=1e-7):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        sigmoid = nn.Sigmoid()
        
        for i in range(len(self.pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos =  -1 * torch.mean(self.pos_weights[i] * y_true[:, i] * torch.log(sigmoid(y_pred[:, i]) + epsilon))
            loss_neg =  -1 * torch.mean(self.neg_weights[i] * (1 - y_true[:, i]) * torch.log(1 -sigmoid( y_pred[:, i]) + epsilon))
            loss += self.pos_weights[i] * (loss_pos + loss_neg)
            # loss = (1 / self.neg_weights[i]) * loss * 0.05
        return loss

def build_lrfn(lr_start=0.000002, lr_max=0.00010, 
               lr_min=0, lr_rampup_epochs=8, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            optimizer.param_groups[0]['lr'] = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            optimizer.param_groups[0]['lr'] = lr_max
        else:
            optimizer.param_groups[0]['lr'] = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return optimizer.param_groups[0]['lr']
    return lrfn

def count_parameters(model): 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

if __name__ == '__main__':

    test_auc_lst = []
    test_acc_lst = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'\ndevice: {device}')
        
    # making empty lists to collect all the losses
    losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}

    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself ! Cool huh ? :D')
    parser.add_argument('--data_path', type = str, default = '.', help = 'This is the path of the training data')
    parser.add_argument('--bs', type = int, default = 64, help = 'batch size')
    parser.add_argument('--loss_func', type = str, default = 'CE', choices = {'BCE', 'FocalLoss', 'CE'}, help = 'loss function')
    parser.add_argument('-r','--resume', default = False ,action = 'store_true') # args.resume will return True if -r or --resume is used in the terminal
    parser.add_argument('--ckpt', type = str, help = 'Path of the ckeckpoint that you wnat to load')
    parser.add_argument('-t','--test', action = 'store_true')   # args.test   will return True if -t or --test   is used in the terminal
    parser.add_argument('--dataset', type = str, default = 'CIFAR10',choices = {'ChexPert','CIFAR10','CIFAR100'}, help = 'This is the path of the training data')
    args = parser.parse_args()
    args.ckpt = 'C:/Users/hb/Desktop/code/FL_distribution_skew/models/stage4_1e-05_12.pth'

    script_start_time = time.time() # tells the total run time of this script

    # mention the path of the data
    data_dir = "C:/Users/hb/Desktop/data/archive"

    # make the datasets
    # random sampling
    random.seed(1996)
    if args.dataset == 'NIH':
        disease = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation']
        indices = list(range(86336))
        random.shuffle(indices)
    elif args.dataset == 'ChexPert':
        disease_chexpert = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture']
        indices = list(range(223414))
        random.shuffle(indices)

    if args.dataset == 'NIH':
        XRayTrain_dataset = XRaysTrainDataset(data_dir, transform = config.transform, indices=indices)
    elif args.dataset == 'ChexPert':
        XRayTrain_dataset = ChexpertTrainDataset(transform = config.transform2, indices=indices)
        XRayTest_dataset = ChexpertTestDataset(transform = config.transform2)
        Total_dataset = torch.utils.data.ConcatDataset([XRayTrain_dataset, XRayTest_dataset])
        train_percentage = 0.8
        train_dataset, temp = torch.utils.data.random_split(Total_dataset, [int(len(Total_dataset)*train_percentage), len(Total_dataset)-int(len(Total_dataset)*train_percentage)])
        discard_percentage = 0.3
        train_dataset, discard = torch.utils.data.random_split(Total_dataset, [int(len(Total_dataset)*discard_percentage), len(Total_dataset)-int(len(Total_dataset)*discard_percentage)])
        test_percentage = 0.5
        val_dataset, test_dataset = torch.utils.data.random_split(temp, [int(len(temp)*test_percentage), len(temp)-int(len(temp)*test_percentage)])
    elif args.dataset == 'CIFAR10':
        CIFAR10Train = CIFAR10TrainDataset()
        test_dataset = CIFAR10TestDataset()
        train_percentage = 0.8
        train_dataset, val_dataset = torch.utils.data.random_split(CIFAR10Train, [int(len(CIFAR10Train)*train_percentage), len(CIFAR10Train)-int(len(CIFAR10Train)*train_percentage)])


    if args.dataset == 'NIH' or args.dataset == 'ChexPert':

        ori_ds_cnt = XRayTrain_dataset.get_ds_cnt()
        # # gan_ds_cnt = GANTrain_dataset.get_ds_cnt()

        # total_ds_cnt = np.array(ori_ds_cnt) + np.array(gan_ds_cnt)
        total_ds_cnt = np.array(ori_ds_cnt)
        print(total_ds_cnt)

        pos_freq = total_ds_cnt / total_ds_cnt.sum()
        neg_freq = 1 - pos_freq

        pos_weights = neg_freq
        neg_weights = pos_freq

        # for effective PNB loss
        raw_pos_freq = total_ds_cnt
        raw_neg_freq = total_ds_cnt.sum() - total_ds_cnt

    # # Plot the disease distribution
    # plt.figure(figsize=(8,4))
    # plt.title('Disease Distribution', fontsize=20)
    # plt.bar(disease_chexpert,total_ds_cnt)
    # plt.tight_layout()
    # plt.gcf().subplots_adjust(bottom=0.40)
    # plt.xticks(rotation = 90)
    # plt.xlabel('Deseases')
    # plt.savefig('disease_distribution_ChexPert(test).png')
    # plt.clf()

    # #Plot the pos neg balancing
    # bar_width = 0.25
    # x = np.arange(len(disease))
    #Before
    # plt.figure(figsize=(8,8))
    # plt.title('Pos Neg Distribution', fontsize=20)
    # plt.bar(x,pos_freq, bar_width)
    # plt.bar(x +  bar_width,neg_freq, bar_width)
    # plt.xticks(x, disease, rotation = 90)
    # plt.savefig('before_balancing.png')
    # plt.clf()
    # #After
    # # plt.figure(figsize=(8,8))
    # plt.title('Pos Neg Distribution', fontsize=20)
    # plt.bar(x,pos_freq*pos_weights,  bar_width)
    # plt.bar(x + bar_width,neg_freq*neg_weights, bar_width)
    # plt.xticks(x, disease, rotation = 90)
    # plt.savefig('after_balancing.png')
    # plt.clf()

    # XRayTrain_dataset = ChestXLoader(0, mode = 'train')
    # XRayTrain_dataset = torch.utils.data.ConcatDataset([XRayTrain_dataset, GANTrain_dataset])
    
    # XRayTest_dataset = XRaysTestDataset(data_dir, transform = config.transform)
    
    # XRayTest_dataset = ChestXTestLoader()
    # XRayTest_dataset = ChestXLoader(mode = 'test')

    print('\n-----Dataset Information-----')
    print('num images in train_dataset   : {}'.format(len(train_dataset)))
    print('num images in val_dataset     : {}'.format(len(val_dataset)))
    print('num images in XRayTest_dataset: {}'.format(len(test_dataset)))
    print('-------------------------------------')

    # make the dataloaders
    batch_size = args.bs # 128 by default
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = not True, num_workers = 4)
    # train2_loader = torch.utils.data.DataLoader(GANTrain_dataset, batch_size = batch_size, shuffle = not True)
 
    if args.loss_func == 'FocalLoss': # by default
        from losses import FocalLoss
        loss_fn = FocalLoss(device = device, gamma = 2.).to(device)
    elif args.loss_func == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    elif args.loss_func == 'PNB':
        loss_fn = weighted_loss(pos_weights, neg_weights)
    elif args.loss_func == 'EPNB':
        loss_fn = Effective_PNB_loss(raw_pos_freq, raw_neg_freq)
    elif args.loss_func == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)

    # import pretrained model
    if args.dataset == 'NIH':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
        model.to(device)
    elif args.dataset == 'ChexPert' or args.dataset == 'CIFAR10':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=10)
        model.to(device)

    for name, param in model.named_parameters(): # all requires_grad by default, are True initially  
        param.requires_grad = True

    if not args.test:
        # checking the layers which are going to be trained (irrespective of args.resume)
        trainable_layers = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                layer_name = str.split(name, '.')[0]
                if layer_name not in trainable_layers: 
                    trainable_layers.append(layer_name)

        print('\nfollowing are the trainable layers...')
        print(trainable_layers)
        print('\nwe have {} Million trainable parameters here in the {} model\n'.format(count_parameters(model), model.__class__.__name__))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0)
    scheduler = build_lrfn()

    # model.load_state_dict(torch.load('C:/Users/hb/Desktop/code/2.TF_to_Torch/Weight/EfficientNetB0.pth'))

    round = 10
    best_auc = 0
    
    if args.dataset == 'CIFAR10':
        fit_CIFAR(device, train_loader, val_loader,    
                                        test_loader, model, loss_fn, 
                                        optimizer, losses_dict,
                                        epochs_till_now = 0, 
                                        log_interval = 25, save_interval = 1,
                                        test_only = True)

    for i in range(1,round+1):

        print('============ EPOCH {}/{} ============'.format(i, round))
        lr = scheduler(i)
        print("Learning Rate : ", lr)

        if args.dataset == 'CIFAR10':
            weight = fit_CIFAR(device, train_loader, val_loader,    
                                        test_loader, model, loss_fn, 
                                        optimizer, losses_dict,
                                        epochs_till_now = i,
                                        log_interval = 25, save_interval = 1,
                                        test_only = False)

    
            model.load_state_dict(weight)
    
            test_auc, test_acc = fit_CIFAR(device, train_loader, val_loader,    
                                        test_loader, model, loss_fn, 
                                        optimizer, losses_dict,
                                        epochs_till_now = i, 
                                        log_interval = 25, save_interval = 1,
                                        test_only = True)

        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(weight,'C:/Users/hb/Desktop/code/2.TF_to_Torch/Weight/CZ_EPNB_Extreme.pth')
    
        print("Current AUC : ", test_auc)
        print("Best AUC : ", best_auc)

        test_acc_lst.append(test_acc)
        test_auc_lst.append(test_auc)

        script_time = time.time() - script_start_time
        m, s = divmod(script_time, 60)
        h, m = divmod(m, 60)
        print('{} h {}m laga poore script me !'.format(int(h), int(m)))

        print("Best ACC: ", max(test_acc_lst))
        print("Best AUC: ", max(test_auc_lst))

    plt.plot(list(range(1,round+1)), test_acc_lst)
    plt.savefig("Accuracy.png")
    plt.clf()
    plt.plot(list(range(1,round+1)), test_auc_lst)
    plt.savefig("AUC.png")
    plt.clf()
