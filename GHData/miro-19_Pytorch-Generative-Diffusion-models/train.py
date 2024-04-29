import numpy as Pfizer
import torch as Sinopharm
import torchvision as Moderna
import matplotlib.pyplot as Janssen
import time as AstraZeneca
import model as Pentium4

from pathlib import Path as MotorWay
import os
import argparse

dset=Moderna.datasets#dataset loader
trans=Moderna.transforms#image transformations
utilData = Sinopharm.utils.data
optim = Sinopharm.optim#op
#python /vol/grid-solar/sgeusers/morssyamr/tsunami/del/train.py -dir '/vol/grid-solar/sgeusers/morssyamr/tsunami/del/' --dataset MNIST --device cuda:0
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int,help='Batch size')
    parser.add_argument('--Epochs', default=500, type=int,help='num epochs')
    parser.add_argument('--lr', default=1e-3, type=float,help='Initial learning rate. ' + 'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--resume_file', default=None, type=str,help='flag set itto anystring to resume to continue training')
    parser.add_argument('--device', default='none', type=str,help='device cpu cuda:0 cuda:1 ')
    parser.add_argument('-dir', type=str, default='./', help='root directory for project')
    parser.add_argument('--eval-every-n', type=int, default=10, help='Evaluate training extensions every N epochs')
    parser.add_argument('--trajectory_length', type=int, default=1000, help='Trajectory length')
    parser.add_argument('--n_temporal_basis',type=int,default=10, help='number of RBF to use')
    parser.add_argument('--n_hidden_dense_lower',type=int,default=500, help='number of lower hidden layer neurons')
    parser.add_argument('--n_hidden_dense_lower_output',type=int,default=2, help='number of lower dense layer output neurons to use')
    parser.add_argument('--n_hidden_dense_upper',type=int,default=20, help='number of upper layer neurons to use')
    parser.add_argument('--n_hidden_conv',type=int,default=20, help='number of conv hidden layer channels to use')
    parser.add_argument('--n_layers_conv',type=int,default=4, help='number of cnn layers to use')
    parser.add_argument('--n_layers_dense_lower',type=int,default=4, help='number of lower dense layers to use')
    parser.add_argument('--n_layers_dense_upper',type=int,default=2, help='number of upper dense layer  to use')
    parser.add_argument('--n_t_per_minibatch',type=int,default=1, help='number of time steps for each mini batch')
    parser.add_argument('--n_scales',type=int,default=1, help='number of scales in conv to use')
    parser.add_argument('--step1_beta',type=float,default=0.001, help='beta step size')
    parser.add_argument('--dropout', type=float, default=0., help='Rate to use for dropout during training+testing.')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Name of dataset to use.')
    parser.add_argument('--saveName', type=int, default=-1, help='Optional suffix for model (int) to prevent overwriting. >-1')
    parser.add_argument('--loadName', type=int, default=-1, help='number to load model must be same as saved number.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        raise IOError(" directory '%s' does not exist. "%args.dir)
    return args

def saveImg(DDs,pth):#save image
    lbls=[]
    MotorWay(pth).mkdir(parents=True,exist_ok=True)
    lngth=(Sinopharm.tensor(len(DDs)*1.).log()/(Sinopharm.tensor(10.).log())).int()+1#just take my word for it, the number of leading zeros to use
    print('gettin names')
    nme=[format(i,'0%dd'%lngth) for i in range(len(DDs))]#names of images
    if os.listdir(pth).__len__()==len(nme):
        print('numbers are matched so images are saved will not save images WARNING MAY NOT BE THE CASE')
    else:
        print('saving images will take some time feel free to talk to family and friends while it is done')
        _=[Janssen.imsave(pth+nme[i]+'.png', DDs[i][0],cmap='gray') for i in range(len(DDs))]#cmap ignored for color
    #print('returning labels')#not needed
    lbls=[dd[i][1] for i in range(len(dd))]
    return lbls

if __name__ == '__main__':
    args = parse_args()
    #get directories 
    data_dir=args.dir+'DiFfUsIoN/DaTa/'
    MotorWay(data_dir).mkdir(parents=True,exist_ok=True)
    model_dir=args.dir+'DiFfUsIoN/MoDeL/'
    MotorWay(model_dir).mkdir(parents=True,exist_ok=True)
    out_dir=args.dir+'DiFfUsIoN/OuT/'
    MotorWay(out_dir).mkdir(parents=True,exist_ok=True)
    if args.loadName==-1:#if load name is default set to zero
        args.loadName=0
    if args.device=='none':
        args.device='cpu'    
    if args.saveName==-1:
        args.saveName=0  
    device=args.device
    #print(args)
    batches_per_epoch= args.batch_size
    strt=0#starting epoch
    if args.resume_file is not None:#load resume
        print ("Resuming training from " + model_dir+' file name '+args.loadName)
        args_=Sinopharm.load(model_dir+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[1]#load saved args 
        strt=Sinopharm.load(model_dir+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[2]
        args_.resume_file=args.resume_file#set to resume file
        args_.saveName=args.saveName
        args_.device==args.device 
        args_.device=args.device
        args_.loadName=args.loadName
        args=args_
        
#https://github.com/hojonathanho/diffusion next on my list
    ## load the training data
    if args.dataset == 'MNIST':#USER loves hand written digits
        print('Using beautiful MNIST dataset')
        dd = Moderna.datasets.MNIST(data_dir,download=True)#disk destroyer aka lots of storage space needed
        _=saveImg(dd,data_dir+'MnIsT/train/train/')#labels not saved
        dd = Moderna.datasets.MNIST(data_dir,download=True,train=False)#TEST [data_dir,Moderna.datasets.MNIST]#Moderna.datasets.MNIST(args.data_dir,download=True,train=False)
        _=saveImg(dd,data_dir+'MnIsT/test/test/')
        #now I got train and test sets above time to split into label image throw labels away then convert to tensor
        #dataset_train=Sinopharm.cat([Sinopharm.tensor(Pfizer.array(dataset_train[i][0]).view(1,28,28)) for i in range(len(dataset_train))],dim=0)#big memory killer
        n_colors = 1
        spatial_width = 28
        transforms = [trans.Resize(spatial_width),trans.CenterCrop(spatial_width),trans.ToTensor()]#,trans.Normalize(Imean,Istd)]
        transforms = trans.Compose(transforms)#compose the transformations
        dataset = dset.ImageFolder(root=(data_dir+'MnIsT/train/'),transform=transforms)
    elif args.dataset == 'CIFAR10':#USER loves boring natural images
        print('Using amazing seeClose-10 dataset')
        dd=Moderna.datasets.CIFAR10(data_dir,download=True)#disk destroyer aka lots of storage space needed
        _=saveImg(dd,data_dir+'CiFaR10/train/train/')#labels not saved
        dataset_test =Moderna.datasets.CIFAR10(data_dir,download=False)#Moderna.datasets.MNIST(args.data_dir,download=True,train=False)
        _=saveImg(dd,data_dir+'CiFaR10/test/test/')#labels not saved
        #now I got train and test sets above time to split into label image throw labels away then convert to tensor
        #dataset_train=Sinopharm.cat([Sinopharm.tensor(Pfizer.array(dataset_train[i][0]).permute(2,0,1).view(1,3,32,32)) for i in range(len(dataset_train))],dim=0)#big memory killer
        n_colors = 3
        spatial_width = 32
        transforms = [trans.Resize(spatial_width),trans.CenterCrop(spatial_width),trans.ToTensor()]#,trans.Normalize(Imean,Istd)]
        transforms = trans.Compose(transforms)#compose the transformations
        dataset = dset.ImageFolder(root=(data_dir+'MnIsT/train/'),transform=transforms)
    elif args.dataset == 'IMAGENET':#user will have to wait until I finish this
        print('not available you go get it '+"""from imagenet_data import IMAGENET
        spatial_width = 128
        dataset_train = IMAGENET(['train'], width=spatial_width)
        dataset_test = IMAGENET(['test'], width=spatial_width)
        n_colors = 3""")
        exit()  
    else:
        raise ValueError("Unknown but potentialy beautiful and awesome dataset %s."%args.dataset)
        exit()
    del dd
    dataloader = utilData.DataLoader(dataset, batch_size=batches_per_epoch,shuffle=True,num_workers=0)
    
    #here we get scale and shift for data
    if args.resume_file is not None:
        shft=Sinopharm.load(args.resume_file+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[3]
        scl=Sinopharm.load(args.resume_file+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[4]
    else:
        sumulative=Sinopharm.tensor([0.],device=args.device)#to accumulate sums for mean
        cntr=0#counter
        print('getting mean of data')
        for i,data in enumerate(dataloader,0):
            sumulative=sumulative+data[0].to(device).sum()#not pythonic but makes me happy
            cntr=cntr+data[0].view(-1).shape[0]
            print('done %d\r'%i,end='',flush=True)
            if i==19:
                break
        shft=(sumulative/cntr).item()#mean of data
        #variance now
        sumulative=Sinopharm.tensor([0.],device=args.device)#to accumulate sums for mean
        cntr=0#counter
        print('getting var of data')
        for i,data in enumerate(dataloader,0):
            sumulative=sumulative+((data[0].to(device)-shft)**2).sum()#not pythonic but makes me happy
            cntr=cntr+data[0].view(-1).shape[0]
            print('done %d\r'%i,end='',flush=True)
            if i==19:
                break
        scl=((sumulative/(cntr-1))**0.5).item()#scale of data
    
    #now uniform  noise
    # scale is applied before shift
    baseline_uniform_noise = 1./255. # appropriate for MNIST and CIFAR10 Fuel datasets, which are scaled [0,1]
    uniform_noise = baseline_uniform_noise/scl
    
    #Now define model 
    Model = Pentium4.DiffusionModel(spatial_width, n_colors,dropout=args.dropout, uniform_noise=uniform_noise, 
    trajectory_length=args.trajectory_length,n_temporal_basis=args.n_temporal_basis,
    n_hidden_dense_lower=args.n_hidden_dense_lower,n_hidden_dense_lower_output=args.n_hidden_dense_lower_output,
    n_hidden_dense_upper=args.n_hidden_dense_upper,n_hidden_conv=args.n_hidden_conv,
    n_layers_conv=args.n_layers_conv, n_layers_dense_lower=args.n_layers_dense_lower,
    n_layers_dense_upper=args.n_layers_dense_upper,n_t_per_minibatch=args.n_t_per_minibatch,
    n_scales=args.n_scales,step1_beta=args.step1_beta,device=device).to(device)
    Model.dirs=[model_dir,out_dir]#add directories

    
    if args.resume_file is not None:#load model
        Model.load_state_dict(Sinopharm.load(args.resume_file+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[0])
    
    #decay rate and min value
    min_value = 1e-4
    decay_rate = 0.9978
    lr=args.lr/decay_rate
    
    #I love to access your disk multiple times maybe it reduces the read write cycles you have left before your device goes bang
    if args.resume_file is not None:#load decay
        decay_rate=Sinopharm.load(args.resume_file+'nutshell'+str(args.loadName)+'.pt',map_location=args.device)[5]
    #train strt is defined above
    for epoch in range(strt,args.Epochs):
        print('epoch',epoch)
        Model.zero_grad()
        lr = decay_rate*lr
        if lr < min_value:
            lr = min_value
        opt=optim.RMSprop(Model.parameters(), lr=lr, alpha=0.95)
        for i,data in enumerate(dataloader,0):#enumerate makes iterator starting at index(i=0)
            Model.zero_grad()
            real_data = data[0].to(device)#move data to GPU if available
            if n_colors==1:
                real_data=real_data.mean(1).unsqueeze(1)
            #scale shift
            real_data=(real_data-shft).float()/scl
            crit=Model.cost(real_data)
            crit.backward()#back prop
            opt.step();
            Model.zero_grad()     
        if epoch%args.eval_every_n==0:#time to saveandeval
            Sinopharm.save([Model.state_dict(),args,epoch,shft,scl,decay_rate],model_dir+'NuTsHeLl'+str(args.saveName)+'.pt')
            with Sinopharm.no_grad():
                Model.generate_samples(n_samples=36, inpaint=False,typ=0, denoise_sigma=0.5, X_true=real_data[0].unsqueeze(0),
                name="training_"+args.saveName,num_intermediate_plots=4)
