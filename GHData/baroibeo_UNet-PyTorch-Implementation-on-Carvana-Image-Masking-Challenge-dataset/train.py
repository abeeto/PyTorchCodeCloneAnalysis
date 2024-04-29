import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import*
from dataset import dataset
from model import UNet
import argparse

def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument("--mask_dir")
    parser.add_argument("--resized_height",default=128,type=int)
    parser.add_argument("--resized_width",default=128,type=int)
    parser.add_argument("--batchsize",default=32,type=int)
    parser.add_argument("--lr",default=1e-4,type=float)
    parser.add_argument("--num_workers",default=2,type=int)
    parser.add_argument("--epochs",default=5,type=int)
    return parser.parse_args()

if __name__=="__main__":
    #Define somethings
    args=arguments()
    resized_height=args.resized_height
    resized_width=args.resized_width
    batchsize=args.batchsize
    lr=args.lr
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device",device)
    num_workers=args.num_workers
    pin_memory=True
    epochs=args.epochs

    img_dir=args.img_dir
    mask_dir=args.mask_dir

    #Train, test split
    img_names=os.listdir(img_dir)
    train_img_names,val_img_names=train_test_split(img_names,test_size=0.1)
    print("num_train: ",len(train_img_names))
    print("num_val: ",len(val_img_names))

    #Data augmentation
    train_transform=get_transform(resized_height,resized_width,train=True)
    val_transform=get_transform(resized_height,resized_width,train=False)

    #Get datasets
    train_ds=dataset(img_dir,mask_dir,train_img_names,train_transform)
    val_ds=dataset(img_dir,mask_dir,val_img_names,val_transform)

    #Get dataloaders
    train_loader=DataLoader(train_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batchsize,num_workers=2,pin_memory=pin_memory,shuffle=True)

    #Get model
    model=UNet(in_channels=3,out_channels=1)

    #Loss function and optimizer
    loss_fn=nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        train_fn(model,train_loader,loss_fn,optimizer,device="cuda")
        checkpoint={
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #test
        train_loss,train_acc,train_dice_score=test_fn(model,train_loader,loss_fn,device)
        val_loss,val_acc,val_dice_score = test_fn(model,val_loader,loss_fn,device)
        print("Epoch : {}/{}, Train Loss :{:.2f}, Val Loss :{:.2f}, Train acc :{:.2f}, Val acc :{:.2f}, Train dice score :{:.2f}, Val dice score :{:.2f}"
            .format(epoch+1,epochs,train_loss,val_loss,train_accuracy,val_accuracy,train_dice_score,val_dice_score))
