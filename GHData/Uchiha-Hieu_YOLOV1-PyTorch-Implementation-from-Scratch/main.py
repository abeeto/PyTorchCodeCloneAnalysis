import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import yolov1
from model.backbones import vgg16
from datasets import geo_shape
import train
import test

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--img_root",type=str,help="image directory")
    parser.add_argument("-j","--json_path",type=str,help="path to labels.json")
    parser.add_argument("-a","--anno_dir",type=str,help="directory contain train/val/test npy idxs")
    parser.add_argument("-c","--cell_size",type=int,default=7,help="Cell Size")
    parser.add_argument("-nb","--num_boxes",type=int,default=2,help="Num predicted boxes per cell")
    parser.add_argument("-b","--batchsize",type=int,default=64,help="Batch size")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="learning rate")
    parser.add_argument("-e","--epochs",type=int,default=100,help="Epochs")
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    img_root = args.img_root
    json_path = args.json_path
    anno_dir = args.anno_dir
    cell_size = args.cell_size
    num_boxes = args.num_boxes
    num_classes = 3
    batchsize = args.batchsize
    lr = args.learning_rate
    epochs = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE : ",device)

    #Get train,val dataset
    train_dataset = geo_shape.GeoShape(img_root,anno_dir,json_path,mode="train")
    val_dataset = geo_shape.GeoShape(img_root,anno_dir,json_path,mode="val")
    print("Nums Train : ",len(train_dataset))
    print("Nums Val : ",len(val_dataset))

    #Get dataloader
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=batchsize,num_workers=4)

    #Get backbone
    backbone = vgg16.VGG16(in_c=1)
    
    #Get model
    model = yolov1.YOLOv1(backbone,cell_size,num_boxes,num_classes).to(device)

    #Get optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(epochs*1/5.),int(epochs*2/5.),int(epochs*3/5.),int(epochs*4/5.),int(epochs*5/5.)],gamma=0.5)

    best_val_iou = 0
    train_loss_hist = []
    val_loss_hist = []
    train_iou_hist = []
    val_iou_hist = []
    for epoch in range(epochs):
        print("Epoch : {}/{}, Learning rate : {}".format(epoch+1,epochs,optimizer.param_groups[0]['lr']))
        train_loss,train_iou = train.run(train_loader,model,optimizer,device,cell_size=cell_size,num_boxes=num_boxes,num_classes=3)
        val_loss,val_iou = test.run(val_loader,model,device,cell_size=cell_size,num_boxes=num_boxes)
        if best_val_iou < val_iou:
            best_val_iou = val_iou
            checkpoints = {
                "state_dict": model.state_dict(),
                "Epoch":epoch,
                "best_val_iou":best_val_iou
            }
            torch.save(checkpoints,"best.pth.tar")
        print("Train loss : {}, Train iou : {}".format(train_loss,train_iou))
        print("Val loss : {},Val iou: {}".format(val_loss,val_iou))
        print("Best Val IOU : ",best_val_iou)
        print("\n")
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_iou_hist.append(train_iou)
        val_iou_hist.append(val_iou)
        lr_scheduler.step()
    
    np.save("train_loss_hist.npy",np.array(train_loss_hist))
    np.save("val_loss_hist.npy",np.array(val_loss_hist))
    np.save("train_iou_hist.npy",np.array(train_iou_hist))
    np.save("val_iou_hist.npy",np.array(val_iou_hist))

