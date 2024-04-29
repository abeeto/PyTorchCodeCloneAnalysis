import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = '/home/oscar/Desktop/20201019_Sentinel2_Dataset_Clean/Images10b/'
dir_mask = '/home/oscar/Desktop/20201019_Sentinel2_Dataset_Clean/Labels3ClassesGrayBKG_2/'
dir_checkpoint = 'checkpoints/'
import os
path = os.getcwd()
print(path)
TRAIN_INDEX = path + "/index/3Classes_Train1.txt" 
TEST_INDEX = path + "/index/3Classes_Test1.txt"



def validation(writer, val_loader, batch_size, global_step, optimizer, EXP_NAME, best_val, save_matrix=False):
    if save_matrix:
        val_score, val_accuracy, val_cm = eval_net(net, val_loader, device, batch_size, save_matrix)
        logging.info('ConfusionMatrix {}'.format(val_cm))
    else:
        val_score, val_accuracy, _ = eval_net(net, val_loader, device, batch_size, save_matrix)

    #scheduler.step(val_score)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

    logging.info('Validation cross entropy: {}'.format(val_score))
    logging.info('GlobalAccuracy: {}'.format(val_accuracy))
    

    writer.add_scalar('Loss/test', val_score, global_step)
    writer.add_scalar('GlobalAccuracy/test', val_accuracy, global_step)

    if (save_matrix):
        np.savetxt("./runs/"+EXP_NAME+"/CM.csv", val_cm, delimiter=";")
    
    # If BEST then Save checkpoint 
    if best_val>= val_score:       
        # Rerun eval to get ConfusionMatrix
        _, _, val_cm = eval_net(net, val_loader, device, batch_size, True)
        np.savetxt("./runs/"+EXP_NAME+"/BEST_CM.csv", val_cm, delimiter=";")

        try:
            os.mkdir("./runs/"+EXP_NAME+"/"+dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

        torch.save(net.state_dict(), "./runs/"+EXP_NAME+"/"+
                dir_checkpoint + 'BEST_MODEL.pth')
        logging.info(f'Model saved !')
        return val_score
    return best_val

def train_net(net,
              device,
              epochs=5,
              batch_size=32,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              EXP_NAME='exp'):#0.5):

    
    trainnames = open(TRAIN_INDEX, 'r').readlines()
    testnames = open(TEST_INDEX, 'r').readlines()
    #print(trainnames)
    traindataset = BasicDataset(trainnames, dir_img, dir_mask, img_scale)
    testdataset = BasicDataset(testnames, dir_img, dir_mask, img_scale)

    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    n_train = len(trainnames)
    n_val = len(testnames)

    #dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = random_split(dataset, [n_train, n_val])
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter("./runs/"+EXP_NAME, comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    BEST_VAL=1000

    logging.info(f'''Starting training:
        Name:            {EXP_NAME}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    optimizer= optim.Adam(net.parameters(), lr=lr, weight_decay=0)#1e-8)#0.0001)
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    weights = [1.1175, 6.0888, 0.4397, 0.9048]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        # VALIDATION at the start of every epoch
        BEST_VAL = validation(writer, val_loader, batch_size, global_step, optimizer, EXP_NAME, BEST_VAL)

        # Start training
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type).squeeze(1)

                masks_pred = net(imgs).squeeze(1)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 1.0)#0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

    # Last validation at the end of the training
    BEST_VAL = validation(writer, val_loader, batch_size, global_step, optimizer, EXP_NAME, BEST_VAL, True)

    # Save checkpoint 
    if save_cp:
        try:
            os.mkdir("./runs/"+EXP_NAME+"/"+dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

        torch.save(net.state_dict(), "./runs/"+EXP_NAME+"/"+
                dir_checkpoint + 'MODEL.pth')
        logging.info(f'Model saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-n', '--name', dest='expname', type=str, default='exp',
                        help='Name of the experimento')               
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=10, n_classes=4, bilinear=False)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    torch.backends.cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  EXP_NAME=args.expname)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
