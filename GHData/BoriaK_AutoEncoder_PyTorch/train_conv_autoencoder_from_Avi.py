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
from torch.utils.tensorboard import SummaryWriter
# This script is for training Auto encoder on unlabeled Images at BMP format

# assumes x is PIL image
from pathlib import Path

parser = argparse.ArgumentParser()
#parser.add_argument('--resume_epoch', type=int, default=19, help='starting epoch')
parser.add_argument('--resume_epoch', type=int, default=None, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--Image_size', type=int, default=128, help='size of the input images')
parser.add_argument('--data', type=str, default=r'./dataSet/DataSet1', help='root directory of the dataset')
parser.add_argument('--lmbd', type=float, default=1 , help='Lambda value for Loss Function')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--root-chkps', type=str, default='./checkpoints', help='checkpoints directory')
parser.add_argument('--logs', type=Path, default="logs")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() and args.cuda else torch.FloatTensor
args.dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() and args.cuda else torch.LongTensor

if not os.path.isdir(args.root_chkps):
    os.mkdir(args.root_chkps)

Loss_Arr = np.zeros((2, args.n_epochs))

root = Path(args.logs)
root.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(root))

def trainAutoEncoder():
    transforms_train = transforms.Compose([
        # transforms.RandomCrop((args.Image_size, args.Image_size)),
        transforms.Resize((args.Image_size, args.Image_size)), 
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
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # get the model using our helper function
    # move model to the right device
    model = AutoEncoder(args.Image_size).to(args.device)

    # for debug
    for i, data in enumerate(data_loader_test):
        debug_output_original = torchvision.transforms.functional.to_pil_image(data.squeeze(0))
        debug_output_original.save(r'./debug_output/orig_' + str(i) + '.bmp')

    # construct an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)

    # and a learning rate scheduler - Optional
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=20,
    #                                                gamma=0.9)

    criterion = nn.L1Loss(reduction='mean').to(args.device)

    if args.resume_epoch is not None:
        checkpoint = torch.load(
            './checkpoints/c_' + str(args.resume_epoch) + '.pth')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # lr_scheduler.state_dict(checkpoint['lr_scheduler_state'])
        start_epoch = args.resume_epoch
        print("LOADED")
    else:
        start_epoch = 0

    torch.backends.cudnn.benchmark = True
    '''train loop'''
    lmbd = args.lmbd  # lambda is a hyper parameter for Rate Distortion loss function
    steps = 0
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        loss_av_train = 0.
        for i, data in enumerate(data_loader):
            data = data.to(args.device).type(args.dtype_float)
            optimizer.zero_grad()
            output = model(data)
            # regularization_loss = 0
            # for param in model.parameters():
            #    regularization_loss += torch.sum(torch.abs(param))
            # loss = criterion(output, data)  + 0.0001*regularization_loss
            # RD Loss Function = MSE + lambda*||y||^2
            # loss = criterion(output[1], data) + lmbd * torch.linalg.norm(output[0], ord=2, dim=1).mean(dim=(0, 1))
            loss = criterion(output[1], data)
            # for debug
            print(str(criterion(output[1], data).item()))
            writer.add_scalar("loss/train", loss.item(), steps)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], steps)            
            # print(str(lmbd * torch.linalg.norm(output[0], ord=2, dim=1).mean(dim=(0, 1)).item()))
            steps += 1
            loss.backward()
            optimizer.step()

            loss_av_train += loss.item()
                                    
            if steps % 10 == 0:
                # grid = torchvision.utils.make_grid(data[].cpu())
                # grid = torchvision.utils.make_grid(output[1].cpu())
                # writer.add_image('images/gt1', grid, steps)
                # run Inference test
                im_grid = torch.empty(0)
                with torch.no_grad():
                    loss_av_test = 0.
                    acc_av_test = 0.
                    model.eval()
                    for i, data in enumerate(data_loader_test):
                        data = data.to(args.device).type(args.dtype_float)
                        output = model(data)
                        loss = criterion(output[1], data)# + lmbd * torch.linalg.norm(output[0], ord=2, dim=1).mean(dim=(0, 1))
                        loss_av_test += loss.item()
                        # torchvision.utils.make_grid(output)
                        # for debug
                        # im_grid = torch.cat((im_grid, output[1].cpu()), dim=0)
                        # writer.add_image('images/output', im_grid, steps)

                        debug_output = torchvision.transforms.functional.to_pil_image(output[1].squeeze(0))
                        debug_output.save(r'./debug_output/' + str(i) + '.bmp')

                    loss_av_test /= len(dataset_valid)

                writer.add_scalar("loss/test", loss_av_test, steps)

        loss_av_train /= len(data_loader) 
        print('epoch:{}/{}, lr={:4f}, '
              'loss-train:{:2f}, '
              'loss-test:{:2f}, '.
              format(epoch, args.n_epochs,
                     optimizer.param_groups[0]["lr"],
                     loss_av_train,
                     loss_av_test))
        if (epoch + 1) % 10 == 0:
            checkpoint = {'epoch': epoch,
                          'model_state': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          }
            torch.save(checkpoint, './checkpoints/c_' +
                       str(epoch) + '.pth')

        # Loss_Arr[0][epoch] = loss_av_train
        # Loss_Arr[1][epoch] = loss_av_test

    # for plot purposes:
    if args.resume_epoch is None:
        start_epoch = 0
    else:
        start_epoch = args.resume_epoch
    plt.figure()
    plt.plot(Loss_Arr[0], range(start_epoch, args.n_epochs), Loss_Arr[1], range(start_epoch, args.n_epochs))
    plt.title("Trainig and Inference Loss over Epochs")
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss")
    plt.legend(['Training Loss', 'Inference Loss'])
    plt.savefig(r'./debug_output/Training_and_Inference_Loss')

if __name__ == "__main__":
    trainAutoEncoder()
