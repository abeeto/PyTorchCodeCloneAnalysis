#inference engine
import torch as Sinopharm
import torchvision as Expert
import time as AstraZeneca
import model as Pentium4
import argparse
import os

dset=Expert.datasets#dataset loader
trans=Expert.transforms#image transformations
utilData = Sinopharm.utils.data
ExpertSystems=Expert.utils

#python /vol/grid-solar/sgeusers/morssyamr/tsunami/del/infer.py -dir '/vol/grid-solar/sgeusers/morssyamr/tsunami/del/' --dataset MNIST --device cuda:0 --loadName 700 --savename -5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='none', type=str,help='device cpu cuda:0 cuda:1 ')
    parser.add_argument('-dir', type=str, default='./', help='root directory for project')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Name of dataset to use.')
    parser.add_argument('--split', type=str, default='train', help='split of dataset to use.')
    parser.add_argument('--BS', type=int, default=36, help='size of batch')
    parser.add_argument('--samples', type=int, default=2, help='number of samples')
    parser.add_argument('--loadName', type=int, default=-1, help='number to use to load model.')
    parser.add_argument('--saveName', type=int, default=-1, help='number to use to save model output.')
    parser.add_argument('--noise', type=float, default=-1, help='noisevar to denoise.')
    parser.add_argument('--inpaint', type=int, default=-1, help='inpaint mask to use. 0 for half image 1 for random default is none')
    parser.add_argument('--examples', default=-1,help='delimited list input ofexamples to do eg "1,5,4,2" if none will generate model samples only', type=str)
    parser.add_argument('--num_intermediate_plots',default=1,type=int,help='How many steps to plot')
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        raise IOError(" directory '%s' does not exist. "%args.dir)
    return args
if __name__ == '__main__':
    args = parse_args()
    
    data_dir=args.dir+'DiFfUsIoN/DaTa/'
    model_dir=args.dir+'DiFfUsIoN/MoDeL/'
    out_dir=args.dir+'DiFfUsIoN/OuT/'
        
    if args.device=='none':
        args.device='cpu'    
    if args.loadName==-1:
        args.loadName=0        
    if args.saveName==-1:
        args.saveName=0    
    if args.noise==-1:
        args.noise=None
    if args.inpaint==-1:
        args.inpaint=None
    if args.examples==-1:
        args.examples=None
    else:
        args.examples = [int(item) for item in args.examples.split(',')]
    device=args.device

   ## load the training data
    if args.dataset == 'MNIST':#USER loves hand written digits
        n_colors = 1
        spatial_width = 28
        Imean = (0.5,0.5,0.5)#tuple of means for each channel
        Istd = (0.5,0.5,0.5)#STD for each channel
        transforms = [trans.Resize(spatial_width),trans.CenterCrop(spatial_width),trans.ToTensor()]#,trans.Normalize(Imean,Istd)]
        transforms = trans.Compose(transforms)#compose the transformations
        if args.split=='train':
            dataset = dset.ImageFolder(root=(data_dir+'MnIsT/train/'),transform=transforms)
        else:
            dataset = dset.ImageFolder(root=(data_dir+'MnIsT/test/'),transform=transforms)
    elif args.dataset == 'CIFAR10':#USER loves boring natural images
        n_colors = 3
        spatial_width = 32
        Imean = (0.5,0.5,0.5)#tuple of means for each channel
        Istd = (0.5,0.5,0.5)#STD for each channel
        transforms = [trans.Resize(spatial_width),trans.CenterCrop(spatial_width),trans.ToTensor()]#,trans.Normalize(Imean,Istd)]
        transforms = trans.Compose(transforms)#compose the transformations
        if args.split=='train':
            dataset = dset.ImageFolder(root=(data_dir+'CiFaR10/train/'),transform=transforms)
        else:
            dataset = dset.ImageFolder(root=(data_dir+'CiFaR10/test/'),transform=transforms)
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
    dataloader = utilData.DataLoader(dataset, batch_size=args.BS,shuffle=False,num_workers=0)
    #now unifroem 
    shft=Sinopharm.load(model_dir+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[3]
    scl=Sinopharm.load(model_dir+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)[4]
    # scale is applied before shift
    baseline_uniform_noise = 1./255. # appropriate for MNIST and CIFAR10 Fuel datasets, which are scaled [0,1]
    uniform_noise = baseline_uniform_noise/scl
    
      
    args_=Sinopharm.load(model_dir+'NuTsHeLl'+str(args.loadName)+'.pt',map_location=args.device)
    #Now define model and optimizer
    Model = Pentium4.DiffusionModel(spatial_width, n_colors,dropout=args_[1].dropout, uniform_noise=uniform_noise, 
    trajectory_length=args_[1].trajectory_length,n_temporal_basis=args_[1].n_temporal_basis,
    n_hidden_dense_lower=args_[1].n_hidden_dense_lower,n_hidden_dense_lower_output=args_[1].n_hidden_dense_lower_output,
    n_hidden_dense_upper=args_[1].n_hidden_dense_upper,n_hidden_conv=args_[1].n_hidden_conv,
    n_layers_conv=args_[1].n_layers_conv, n_layers_dense_lower=args_[1].n_layers_dense_lower,
    n_layers_dense_upper=args_[1].n_layers_dense_upper,n_t_per_minibatch=args_[1].n_t_per_minibatch,
    n_scales=args_[1].n_scales,step1_beta=args_[1].step1_beta,device=device).to(device)
    Model.dirs=[model_dir,out_dir]#add directories
    Model.load_state_dict(args_[0])
    Model.to(args.device)

    Batch=next(iter(dataloader))[0]
    if n_colors==1:
        Batch=Batch.mean(1).unsqueeze(1)
    with Sinopharm.no_grad(): 
        sigma=args.noise
        msk=args.inpaint
        inp=False
        if not msk==None:
            inp=True
        if args.examples==None:
            msk=None
            sigma=None
            inp=False
            Model.generate_samples(n_samples=args.samples, inpaint=inp,typ=msk, denoise_sigma=sigma, X_true=None,
                name="Inference example_"+str(args.saveName),num_intermediate_plots=args.num_intermediate_plots)
        else:
            for i in range(Batch.shape[0]):
                if i in args.examples:
                    Model.generate_samples(n_samples=args.samples, inpaint=inp,typ=msk, denoise_sigma=sigma, X_true=Batch[i].unsqueeze(0).to(args.device),
                name=("Inference example_%d_"%i)+str(args.saveName),num_intermediate_plots=args.num_intermediate_plots)
