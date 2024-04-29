import argparse
import os
import torch
import torch.nn as nn
from config import Config
from dataloader import get_dataloader
from get_model import get_model
from train import train
from test import test
from utils import save_checkpoints

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_mode",type=int,default=0,help="0 : Cifar10, 1 : Cifar100")
    parser.add_argument("-m","--model_type",type=str,default="vgg11",help="Model type")
    parser.add_argument("-lr","--learning_rate",type=float,default=1e-2,help="learning rate")
    parser.add_argument("-lrd","--learning_rate_decay",type=float,default=0.1,help="learning rate decay")
    parser.add_argument("-p","--pretrained_model_path",type=str,default=None,help="pretrained model path (if used)")
    parser.add_argument("-o","--output_model_folder",type=str,default="./model_folder")
    parser.add_argument("-e","--epochs",type=int,default=100,help="Epochs for training")
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    if args.dataset_mode !=0 and args.dataset_mode !=1:
        raise ValueError("dataset mode should be 0 (CIFAR 10) or 1 (CIFAR 100) ")
    
    if args.model_type not in Config.MODEL_TYPE.value:
        raise ValueError("Model Type should be in : ",Config.MODEL_TYPE.value)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10 if args.dataset_mode == 0 else 100

    #Get dataset
    train_loader = get_dataloader(args.dataset_mode,isTrain = True)
    test_loader = get_dataloader(args.dataset_mode,isTrain = False)

    #Get model
    model = get_model(args.model_type,num_classes = num_classes).to(device)

    #Get loss
    criterion = nn.CrossEntropyLoss()

    #Get optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=Config.EPOCH_LR_SCHEDULER.value,gamma = args.learning_rate_decay)
    
    #train and test
    best_test_acc = 0
    train_accuracies = {}
    train_losses = {}
    test_losses = {}
    test_accuracies = {}
    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate : {}".format(epoch+1,args.epochs,optimizer.param_groups[0]['lr']))
        #Train
        if (epoch+1) % Config.EVAL_TRAIN_STEP.value == 0 or epoch==0:
            train_loss,train_acc = train(train_loader,model,criterion,optimizer,device,isEval=True)
            train_accuracies["Epoch_"+str(epoch+1)] = train_acc
            train_losses["Epoch_"+str(epoch+1)] = train_loss
        
        else:
            train_loss = train(train_loader,model,criterion,optimizer,device)
        
        #Test
        test_loss,test_acc = test(test_loader,model,criterion,device)
        test_accuracies["Epoch_"+str(epoch+1)] = test_acc
        test_losses["Epoch_"+str(epoch+1)] = test_loss

        if (epoch+1) % Config.EVAL_TRAIN_STEP.value == 0 or epoch == 0:
            loss_acc_checkpoints = {
                "train_losses" : train_losses,
                "train_accuracies": train_accuracies,
                "test_losses" : test_losses,
                "test_accuracies" : test_accuracies
            }
            save_checkpoints(loss_acc_checkpoints,args.output_model_folder,args.model_type+"_loss_acc.pth.tar")
            print("Train Loss : {},Train Acc :{}".format(train_loss,train_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoints = {
                "state_dict": model.state_dict(),
                "Epoch":epoch,
                "test_losses":test_loss,
                "test_acc":test_acc
            }
            save_checkpoints(checkpoints,args.output_model_folder,args.model_type+"_best.pth.tar")

        print("test_acc : {} , best test acc : {}".format(test_acc,best_test_acc))

        lr_scheduler.step()
