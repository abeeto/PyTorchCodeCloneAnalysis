
import os
import random
import torch
#root
import sys
sys.path.append(os.path.join(os.getcwd(),"Perera2019-OCGAN"))
import options

import utility
from utility import *
from metric import *
from networks import *
import dataset
import dataloader

import torch

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from tqdm import tqdm
from tqdm import trange
import torchvision

from sklearn.metrics import roc_auc_score, roc_curve, auc
import shutil
import time
import math
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image

def main(opt):
    args = opt   
    state = {k: v for k, v in args._get_kwargs()}
    latent = args.latent
    #Use Cuda
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
    use_cuda=torch.cuda.is_available()
    exp = str(args.exp_name)
    batch = args.batch
    #Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1,10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    global best_acc
    start_epoch = args.start_epoch
    if not os.path.exists(args.dataroot):
        os.mkdir(args.dataroot)

    if not "training" in os.listdir(args.dataroot):
        print("data split")
        if args.protocol==1:
            print("Protocol 1111111 dataSplit")
            dataset.dataspread()
        else:
            print("Protocol 2222222 dataSplit")
            dataset.dataspread2()
    print("Preparing dataset")
    dataloading=dataloader.set_data()

    Tensor = torch.cuda.FloatTensor
    
    print("creating model")
    title = "OCGAN"
    
    enc = get_encoder(args).cuda()
    dec = get_decoder(args).cuda()

    #load origianal weights
    enc.apply(weights_init)
    dec.apply(weights_init)

    
    model = torch.nn.DataParallel(enc).cuda()
    cudnn.benchmark = True


    print("creating optimizer")

    criterion_ce = torch.nn.BCELoss(size_average=True).cuda()
    criterion_ae = nn.MSELoss(size_average=True).cuda()


    #Adam optimizer
    optimizer_en = optim.Adam(enc.parameters(),lr=args.lr, betas=(0.5,0.999)) #encoder
    optimizer_de = optim.Adam(dec.parameters(),lr=args.lr, betas=(0.5,0.999)) #decoder 


    if args.resume:
        print("Resuming from checkpoint")
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found'
        args.checkpoint = os.path.dirname(args.resume)

        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = chechpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title= title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate','Train Loss','Valid Loss', 'Test AUC'])


    for epoch in tqdm(range(start_epoch, args.epochs)):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # training 
        train_loss=train(args, dataloading['train'], enc, dec, optimizer_en, optimizer_de,   
                         criterion_ae,criterion_ce,
                         Tensor, epoch, use_cuda)
        #validation
        valid_loss = valid(args, dataloading['valid'],enc,dec,epoch, use_cuda)
        # testing
        test_acc = test(args, dataloading['test'], enc, dec, epoch,use_cuda)

        #append logger file
        print(valid_loss)
        logger.append([state['lr'], train_loss, valid_loss ,test_acc])
        #save model
        is_best = valid_loss < best_acc
        best_acc=min(valid_loss, best_acc)
        
        save_checkpoint({
                         'epoch': epoch+1,
                         'state_dict':enc.state_dict(),
                         'loss': valid_loss,
                         'best_loss': best_acc,
                         }, is_best,checkpoint=args.checkpoint, filename = 'enc_model.pth.tar')
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': dec.state_dict(),
                    'loss': valid_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='dec_model.pth.tar')
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint,'log.png'))

    print("Best acc :")
    print(best_acc)

def train(args, trainloader,enc,dec,optimizer_en, optimizer_de,          
           criterion_ae, criterion_ce, Tensor, epoch, use_cuda):
    enc.train()
    dec.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    isize=args.isize
    csize=args.cropsize
    batch = args.batch
    latent = args.latent
    exp = args.exp_name
    end=time.time()
    bat=0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        
        data_time.update(time.time()-end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        



        
        u = np.random.uniform(-1,1, (batch, latent,1,1))
        l2 = torch.from_numpy(u).float().cuda()
        l2 = torch.tanh(l2)
        dec_l2 = dec(l2,args)
 
        n= torch.normal(0,0.2,size=(batch, 3, csize , csize)).cuda()
        inputs_noise = inputs + n
        l1 = enc(inputs_noise)

        ######  update ae 
        Xh = dec(l1,args)

        loss_mse = criterion_ae(Xh,inputs) #errR
        
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()

        loss_ae_all = loss_mse
        loss_ae_all.backward()

        optimizer_en.step()
        optimizer_de.step()

        losses.update(loss_ae_all.item(), inputs.size(0))



       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        bat+=batch_idx



    if epoch % 5 ==0:
        recon=dec(enc(inputs),args)
        recon=recon.cpu().data
        recon=recon/2 +0.5
        inputs_noise=inputs_noise.cpu().data
        inputs_noise=inputs_noise/2+0.5
        inputs=inputs.cpu().data
        inputs=inputs/2+0.5
        if not os.path.exists('./result'):
            os.mkdir('./result')
        if not os.path.isdir(os.path.join("result",exp,"real")):
            utility.mkdir_p(os.path.join("result",exp,"real"))
        if not os.path.isdir(os.path.join("result",exp,"fake")):
            utility.mkdir_p(os.path.join("result",exp,"fake"))
        if not os.path.isdir(os.path.join("result",exp,"real_noise")):
            utility.mkdir_p(os.path.join("result",exp,"real_noise"))

        fake_name = 'fake'+exp+'_'+str(epoch)+'.png'
        real_name = 'real'+exp+'_'+str(epoch)+'.png'
        real_noise_name = 'real'+exp+'_'+str(epoch)+'.png'
        save_image(recon, os.path.join(".", "result",exp,"fake",fake_name))
        save_image(inputs, os.path.join(".", "result",exp,"real",real_name))
        save_image(inputs_noise, os.path.join(".", "result",exp,"real_noise",real_noise_name))

    return losses.avg

def valid(args,validloader, enc, dec,epoch, use_cuda):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    vloss = AverageMeter()


    # switch to evaluate mode
    enc.eval()
    dec.eval()

    end = time.time()
    batch = args.batch
    isize = args.isize
    csize = args.cropsize
    exp = args.exp_name
    for batch_idx, (inputs, targets) in enumerate(validloader):
        

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
              
            recon = dec(enc(inputs),args)
            scores = torch.mean(torch.pow((inputs - recon), 2),dim=[0,1,2,3])
        vloss.update(scores.cpu().detach().numpy())
    return vloss.avg


def test(args,testloader, enc, dec, epoch, use_cuda):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    enc.eval()
    dec.eval()

    end = time.time()
    batch = args.batch
    isize = args.isize
    csize = args.cropsize
    exp = args.exp_name
    for batch_idx, (inputs, targets) in enumerate(testloader):
        

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
        
      
            recon = dec(enc(inputs),args)
            scores = torch.mean(torch.pow((inputs - recon), 2),dim=[1,2,3])
            prec1 = roc_auc_score(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())
            
        top1.update(prec1, inputs.size(0))



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(top1.avg)  
    recon=recon.cpu().data
    recon=recon/2+0.5
    inputs=inputs.cpu().data
    inputs=inputs/2+0.5
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.isdir(os.path.join("result",exp,"test_real")):
        utility.mkdir_p(os.path.join("result",exp,"test_real"))
    if not os.path.isdir(os.path.join("result",exp,"test_fake")):
        utility.mkdir_p(os.path.join("result",exp,"test_fake"))
    t_fake_name = 'fake'+exp+'_'+str(epoch)+'.png'
    t_real_name = 'real'+exp+'_'+str(epoch)+'.png'
    save_image(recon, os.path.join(".", "result",exp,"test_fake",t_fake_name))
    save_image(inputs, os.path.join(".", "result",exp,"test_real",t_real_name))
        

        


    return top1.avg



def save_checkpoint(state, is_best, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    opt = options.train_options()
    best_acc = 1000
    main(opt)

