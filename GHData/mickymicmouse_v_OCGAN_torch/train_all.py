
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
    disc_v = get_disc_visual().cuda()
    disc_l = get_disc_latent(args).cuda() 
    cl = get_classifier().cuda()

    #load origianal weights
    disc_v.apply(weights_init)
    cl.apply(weights_init)
    enc.apply(weights_init)
    dec.apply(weights_init)
    disc_l.apply(weights_init)
    
    model = torch.nn.DataParallel(enc).cuda()
    cudnn.benchmark = True


    print("creating optimizer")

    criterion_ce = torch.nn.BCELoss(size_average=True).cuda()
    criterion_ae = nn.MSELoss(size_average=True).cuda()



    #Adam optimizer
    optimizer_en = optim.Adam(enc.parameters(),lr=args.lr, betas=(0.5,0.999)) #encoder
    optimizer_de = optim.Adam(dec.parameters(),lr=args.lr, betas=(0.5,0.999)) #decoder
    optimizer_dl = optim.Adam(disc_l.parameters(),lr=args.lr, betas=(0.5,0.999)) #latent discriminator 
    optimizer_dv = optim.Adam(disc_v.parameters(),lr=args.lr, betas=(0.5,0.999)) #visual discriminator
    optimizer_c = optim.Adam(cl.parameters(),lr=args.lr, betas=(0.5,0.999)) #classifier

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
        train_loss=train(args, dataloading['train'], enc, dec, cl, disc_l, disc_v,
                         optimizer_en, optimizer_de, optimizer_c, optimizer_dl,optimizer_dv,
                         criterion_ae,criterion_ce,
                         Tensor, epoch, use_cuda)
        #validation
        valid_loss = valid(args, dataloading['valid'],enc,dec,cl,disc_l, disc_v, epoch, use_cuda)
        # testing
        test_acc = test(args, dataloading['test'], enc, dec, cl, disc_l, disc_v, epoch,use_cuda)

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
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': cl.state_dict(),
                    'loss': valid_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='cl_model.pth.tar')
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': disc_l.state_dict(),
                    'loss': valid_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='disc_l_model.pth.tar')
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': disc_v.state_dict(),
                    'loss': valid_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='disc_v_model.pth.tar') 
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint,'log.png'))

    print("Best acc :")
    print(best_acc)

def train(args, trainloader, enc, dec, cl, disc_l, disc_v,
          optimizer_en, optimizer_de, optimizer_c, optimizer_dl, optimizer_dv,
          criterion_ae, criterion_ce, Tensor, epoch, use_cuda):
    enc.train()
    dec.train()
    cl.train()
    disc_l.train()
    disc_v.train()
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


        # classifier update

        logits_C_l1 = cl(dec(l1,args))
        logits_C_l2 = cl(dec(l2,args))


        valid_logits_C_l1 = Variable(Tensor(logits_C_l1.shape[0], 1).fill_(1.0), requires_grad=False)
        fake_logits_C_l2 = Variable(Tensor(logits_C_l2.shape[0], 1).fill_(0.0), requires_grad=False)


        loss_cl_l1 = criterion_ce(logits_C_l1,valid_logits_C_l1)
        loss_cl_l2 = criterion_ce(logits_C_l2,fake_logits_C_l2)

        loss_cl = loss_cl_l1 + loss_cl_l2


        optimizer_c.zero_grad()
        loss_cl.backward(retain_graph=True)
        optimizer_c.step()



        #Discriminator update

        logits_Dl_l1 = disc_l(l1)
        logits_Dl_l2 = disc_l(l2)


        dl_logits_DL_l1 = Variable(Tensor(logits_Dl_l1.shape[0], 1).fill_(0.0), requires_grad=False)
        dl_logits_DL_l2 = Variable(Tensor(logits_Dl_l2.shape[0], 1).fill_(1.0), requires_grad=False)
        loss_dl_1 = criterion_ce(logits_Dl_l1 , dl_logits_DL_l1)
        loss_dl_2 = criterion_ce(logits_Dl_l2 , dl_logits_DL_l2)
        loss_dl = loss_dl_1 + loss_dl_2

        # visual Discriminator l1 = real l2 = random
        logits_Dv_X = disc_v(inputs)
        logits_Dv_l2 = disc_v(dec(l2,args))

        dv_logits_Dv_X = Variable(Tensor(logits_Dv_X.shape[0], 1).fill_(1.0), requires_grad=False)
        dv_logits_Dv_l2 = Variable(Tensor(logits_Dv_l2.shape[0], 1).fill_(0.0), requires_grad=False)

        optimizer_dl.zero_grad()
        optimizer_dv.zero_grad()
        
        loss_dv_1 = criterion_ce(logits_Dv_X,dv_logits_Dv_X)
        loss_dv_2 = criterion_ce(logits_Dv_l2,dv_logits_Dv_l2)
        loss_dv = loss_dv_1 + loss_dv_2


        loss_second=loss_dv+loss_dl     
        loss_second.backward(retain_graph=True)


        optimizer_dv.step()
        optimizer_dl.step()


        for i in range(5):
            logits_C_l2_mine = cl(dec(l2,args))
            zeros_logits_C_l2_mine = Variable(Tensor(logits_C_l2_mine.shape[0], 1).fill_(1.0), requires_grad=False)
            loss_C_l2_mine = criterion_ce(logits_C_l2_mine,zeros_logits_C_l2_mine)
            optimizer_c.zero_grad()
            loss_C_l2_mine.backward()
            optimizer_c.step()

        ######  update ae 


        Xh = dec(l1,args)

        loss_mse = criterion_ae(Xh,inputs) #errR


        
        logits_Dl_l1=disc_l(l1)
        #logits_Dl_l2=disc_l(l2)
        ones_logits_Dl_l1 = Variable(Tensor(logits_Dl_l1.shape[0], 1).fill_(1.0), requires_grad=False)
        # zeros_logits_Dl_l2 = Variable(Tensor(logits_Dl_l2.shape[0], 1).fill_(0.0), requires_grad=False)
        loss_AE_l = criterion_ce(logits_Dl_l1,ones_logits_Dl_l1)


        logits_Dv_l1_mine = disc_v(dec(l2,args))
        # logits_Dv_l2_mine = disc_v(inputs)
        ones_logits_Dv_l1_mine = Variable(Tensor(logits_Dv_l1_mine.shape[0], 1).fill_(1.0), requires_grad=False)
        # zeros_logits_Dv_l2_mine = Variable(Tensor(logits_Dv_l2_mine.shape[0], 1).fill_(0.0), requires_grad=False)
        loss_ae_v = criterion_ce(logits_Dv_l1_mine, ones_logits_Dv_l1_mine)

        optimizer_en.zero_grad()
        optimizer_de.zero_grad()

        loss_ae_all = 10.0 * loss_mse + loss_ae_v + loss_AE_l

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

def valid(args,validloader, enc, dec,cl,disc_l,disc_v, epoch, use_cuda):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    vloss = AverageMeter()


    # switch to evaluate mode
    enc.eval()
    dec.eval()
    cl.eval()
    disc_l.eval()
    disc_v.eval()

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


def test(args,testloader, enc, dec,cl,disc_l,disc_v, epoch, use_cuda):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top1.reset()

    # switch to evaluate mode
    enc.eval()
    dec.eval()
    cl.eval()
    disc_l.eval()
    disc_v.eval()

    end = time.time()
    batch = args.batch
    isize = args.isize
    csize = args.cropsize
    exp = args.exp_name
    fpr_list = []
    tpr_list=[]
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

