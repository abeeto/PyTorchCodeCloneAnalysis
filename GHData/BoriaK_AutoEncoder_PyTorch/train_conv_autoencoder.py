import os
import sys
import argparse
import matplotlib as plt
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.utils
import matplotlib.pyplot as plt
from models import AutoEncoder
from PIL import Image
from datasets import DataSet1
import torchvision.transforms as transforms
import glob

# This script is for training Auto encoder on unlabeled Images at BMP format

# assumes x is PIL image


parser = argparse.ArgumentParser()
parser.add_argument('--resume_epoch', type=int, default=None, help='resume epoch value')
parser.add_argument('--load_training_phase', type=str, default='c_', help='c_ - regular, c1_ - w. PCA Rotation, c2_ - '
                                                                          'w. quantization noise, c_3_ - w. RD Loss, '
                                                                          'f_ - final version')
parser.add_argument('--save_training_phase', type=str, default=None, help='c_ - regular, c1_ - w. PCA Rotation, c2_ - '
                                                                          'w. quantization noise, c_3_ - w. RD Loss f_ '
                                                                          '- final version')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--Image_size', type=int, default=128, help='size of the input images')
parser.add_argument('--data', type=str, default=r'./dataSet/DataSet1', help='root directory of the dataset')
parser.add_argument('--lmbd', type=float, default=1, help='Lambda value for Loss Function')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--root-chkps', type=str, default='./checkpoints', help='checkpoints directory')
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

# args.dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() and args.cuda else torch.FloatTensor
# args.dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() and args.cuda else torch.LongTensor


if not os.path.isdir(args.root_chkps):
    os.mkdir(args.root_chkps)

Loss_Arr = np.zeros((2, args.n_epochs))


def trainAutoEncoder():
    transforms_train = transforms.Compose([
        # transforms.RandomCrop((args.Image_size, args.Image_size)),
        transforms.Resize((args.Image_size, args.Image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        # transforms.RandomCrop((args.Image_size, args.Image_size)),
        transforms.Resize((args.Image_size, args.Image_size)),
        transforms.ToTensor()
    ])

    dataset_train = DataSet1(args.data, mode='Train', transforms_=transforms_train)
    dataset_valid = DataSet1(args.data, mode='Validation', transforms_=transforms_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=0)

    # get the model using our helper function
    # move model to the right device
    model = AutoEncoder(args.Image_size).to(args.device)

    # for debug
    for i, data in enumerate(data_loader_test):
        debug_output_original = torchvision.transforms.functional.to_pil_image(data.squeeze(0))
        debug_output_original.save(r'./debug_output/orig_' + str(i) + '.bmp')

    # construct an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4)

    # and a learning rate scheduler - Optional
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=20,
    #                                                gamma=0.9)

    # criterion = nn.MSELoss(reduction='mean').to(args.device)
    criterion = nn.L1Loss(reduction='mean').to(args.device)

    # args.resume_epoch = 2

    if args.save_training_phase is None:
        save_training_phase = args.load_training_phase
    else:
        save_training_phase = args.save_training_phase

    if args.resume_epoch is not None:
        checkpoint = torch.load(
            './checkpoints/' + args.load_training_phase + str(args.resume_epoch) + '.pth')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # lr_scheduler.state_dict(checkpoint['lr_scheduler_state'])
        start_epoch = args.resume_epoch
        print("Checkpoint " + args.load_training_phase + str(args.resume_epoch) + " is LOADED")
    else:
        start_epoch = 0

    '''train loop'''
    lmbd = args.lmbd  # lambda is a hyper parameter for Rate Distortion loss function
    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        model.train()
        loss_av_train = 0.
        for i, data in enumerate(data_loader):
            data = data.to(args.device)  # .type(args.dtype_float)
            optimizer.zero_grad()
            output = model(data)
            # regularization_loss = 0
            # for param in model.parameters():
            #    regularization_loss += torch.sum(torch.abs(param))
            # loss = criterion(output, data)  + 0.0001*regularization_loss
            # RD Loss Function = MSE + lambda*||y||^2
            loss = criterion(output[1], data) + lmbd * torch.linalg.norm(output[0], ord=2, dim=1).mean(dim=(0, 1))
            # loss = criterion(output[1], data)
            # for debug
            print(str(criterion(output[1], data).item()))
            print(str(lmbd * torch.linalg.norm(output[0], ord=2, dim=1).mean(dim=(0, 1)).item()))

            loss.backward()
            optimizer.step()

            loss_av_train += loss.item()

        loss_av_train /= len(data_loader)

        # run Inference test
        with torch.no_grad():
            loss_av_test = 0.
            acc_av_test = 0.
            model.eval()
            for i, data in enumerate(data_loader_test):
                data = data.to(args.device)  # .type(args.dtype_float)
                output = model(data)
                loss = criterion(output[1], data) + lmbd * torch.linalg.norm(output[0], ord=2, dim=1).mean(dim=(0, 1))
                # loss = criterion(output[1], data)
                loss_av_test += loss.item()
                # torchvision.utils.make_grid(output)
                # for debug
                debug_output = torchvision.transforms.functional.to_pil_image(output[1].squeeze(0))
                debug_output.save(r'./debug_output/' + str(i) + '.bmp')

        loss_av_test /= len(dataset_valid)

        print('epoch:{}/{}, lr={:4f}, '
              'loss-train:{:2f}, '
              'loss-test:{:2f}, '.
              format(epoch, start_epoch + args.n_epochs,
                     optimizer.param_groups[0]["lr"],
                     loss_av_train,
                     loss_av_test))
        # if (epoch + 1) % 10 == 0:
        #     checkpoint = {'epoch': epoch,
        #                   'model_state': model.state_dict(),
        #                   'optimizer_state': optimizer.state_dict(),
        #                   }
        #     torch.save(checkpoint, './checkpoints/c_' +
        #                str(epoch) + '.pth')

        if epoch == 0:
            best_loss = loss_av_test
            checkpoint = {'epoch': epoch,
                          'model_state': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          }
            torch.save(checkpoint, './checkpoints/' + save_training_phase + str(epoch) + '.pth')
            print("Checkpoint " + save_training_phase + str(epoch) + " is SAVED")
        elif (epoch - start_epoch) == 0:
            best_loss = loss_av_test
        else:
            if loss_av_test < best_loss:
                best_loss = loss_av_test
                files = glob.glob(r'./checkpoints/*')
                for items in files:
                    os.remove(items)
                checkpoint = {'epoch': epoch,
                              'model_state': model.state_dict(),
                              'optimizer_state': optimizer.state_dict(),
                              }
                torch.save(checkpoint, './checkpoints/' + save_training_phase + str(epoch) + '.pth')
                print("Checkpoint " + save_training_phase + str(epoch) + " is SAVED")
                # torch.save(checkpoint, './checkpoints/BEST.pth')

        Loss_Arr[0][epoch - start_epoch] = loss_av_train
        Loss_Arr[1][epoch - start_epoch] = loss_av_test

    # for plot purposes:
    plt.figure()
    plt.plot(range(start_epoch, start_epoch + args.n_epochs), Loss_Arr[0],
             range(start_epoch, start_epoch + args.n_epochs),
             Loss_Arr[1])
    plt.title("Trainig and Inference Loss over Epochs")
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss")
    plt.legend(['Training Loss', 'Inference Loss'])
    plt.savefig(r'./debug_output/Training_and_Inference_Loss')


if __name__ == "__main__":
    trainAutoEncoder()
