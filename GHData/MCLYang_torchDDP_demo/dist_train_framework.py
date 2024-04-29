from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('/')
from argparse import ArgumentParser
import os
import random
from tqdm import tqdm
import numpy as np
import pdb
from glob import glob
import pandas as pd
from metrics_manager import metrics_manager
from pathlib import Path
import time
import wandb
from collections import OrderedDict
import random
from BigredDataSet import BigredDataSet
from kornia.utils.metrics import mean_iou,confusion_matrix
import pandas as pd
import importlib
import shutil

from apex import amp
import apex
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.parallel

def opt_global_inti():
    setSeed(10)

    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='some_name')
    parser.add_argument('--notification_email', type=str, default='will@email.com')
    parser.add_argument('--dataset_root', type=str, default='../bigRed_h5_pointnet', help="dataset path")
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)

    parser.add_argument('--phase', type=str,default='Train' ,help=['Train,Test'])
    parser.add_argument('--num_points', type=int,default=20000 ,help="use feature transform")

    parser.add_argument('--load_pretrain', type=str,default='',help="root load_pretrain")
    parser.add_argument('--synchonization', type=str,default='BN' ,help="[BN,BN_syn,Instance]")
    parser.add_argument('--tol_stop', type=float,default=1e-5 ,help="early stop for loss")


    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
    parser.add_argument('--num_channel', type=int,default=4,help="num_channel")
    parser.add_argument('--epoch_max', type=int,default=1,help="epoch_max")
    parser.add_argument('--model', type=str,default='pointnet_ring_light' ,help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")
    parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'),default=True ,help="is task for debugging?False for load entire dataset")
    parser.add_argument('--including_ring', type=lambda x: (str(x).lower() == 'true'),default=True ,help="Including the laserID?")

    # parser.add_argument('--wd_project', type=str,default="point_ring_light_Dist VS point_ring_light" ,help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")
    parser.add_argument('--wd_project', type=str,default="debug",help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")

    #multiprocess
    parser.add_argument('--apex', type=lambda x: (str(x).lower() == 'true'),default=True ,help="use apex mix precision?")
    parser.add_argument('--opt_level', default='O2',type=str, metavar='N')
    parser.add_argument('--num_process_pgpu', default=2,type=int, metavar='N',help="How many process for each gpu?")
    parser.add_argument('--num_gpu', type=int,default=2,help="num_gpu")
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = parser.parse_args()
    return args


def save_model(package,root):
    torch.save(package,root)

def setSeed(seed = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def convert_state_dict(state_dict):
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def visualize_wandb(points,pred,target):
    # points [B,N,C]->[B*N,C]
    # pred,target [B,N,1]->[B*N,1]
    points = points.view(-1,5).numpy()
    pred = pred.view(-1,1).numpy()
    target = target.view(-1,1).numpy()
    points_gt =np.concatenate((points[:,[0,1,2]],target),axis=1)
    points_pd =np.concatenate((points[:,[0,1,2]],pred),axis=1)
    wandb.log({"Ground_truth": wandb.Object3D(points_gt)})
    wandb.log({"Prediction": wandb.Object3D(points_pd)})


class tag_getter(object):
    def __init__(self,file_dict):
        self.sorted_keys = np.array(sorted(file_dict.keys()))
        self.file_dict = file_dict
    def get_difficulty_location_isSingle(self,j):
        temp_arr = self.sorted_keys<=j
        index_for_keys = sum(temp_arr)
        _key = self.sorted_keys[index_for_keys-1]
        file_name = self.file_dict[_key]
        file_name = file_name[:-3]
        difficulty,location,isSingle = file_name.split("_")
        return(difficulty,location,isSingle,file_name)


def generate_report(summery_dict,package):
    save_sheet=[]
    save_sheet.append(['name',package['name']])
    save_sheet.append(['validation_miou',package['Validation_ave_miou']])
    save_sheet.append(['test_miou',summery_dict['Miou']])
    save_sheet.append(['Biou',summery_dict['Biou']])
    save_sheet.append(['Fiou',summery_dict['Fiou']])
    save_sheet.append(['time_complexicity(f/s)',summery_dict['time_complexicity']])
    save_sheet.append(['storage_complexicity',summery_dict['storage_complexicity']])
    save_sheet.append(['number_channel',package['num_channel']])
    save_sheet.append(['Date',package['time']])
    save_sheet.append(['Training-Validation-Testing','0.7-0.9-1'])
    
    for name in summery_dict:
        if(name!='Miou' 
            and name!='storage_complexicity'
            and name!='time_complexicity'
            and name!='Biou'
            and name!='Fiou'
            ):
            save_sheet.append([name,summery_dict[name]])
        print(name+': %2f' % summery_dict[name])
    # pdb.set_trace()
    save_sheet.append(['para',''])
    
    f = pd.DataFrame(save_sheet)
    f.to_csv('testReport.csv',index=False,header=None)


def load_pretrained(opt):
    print('----------------------loading Pretrained----------------------')
    pretrained_model_path = os.path.join(opt.load_pretrain,'best_model.pth')
    package = torch.load(pretrained_model_path)
    para_state_dict = package['state_dict']
    opt.num_channel = package['num_channel']
    opt.time = package['time'] 
    opt.epoch_ckpt = package['epoch']
    opt.val_miou = package['Validation_ave_miou'] 
    state_dict = convert_state_dict(para_state_dict)
    ckpt_,ckpt_file_name  = opt.load_pretrain.split("/")
    module_name = ckpt_+'.'+ckpt_file_name+'.'+'model'
    MODEL = importlib.import_module(module_name)
    model = MODEL.get_model(input_channel = opt.num_channel)
    Model_Specification = MODEL.get_model_name(input_channel = opt.num_channel)
    f_loss = MODEL.get_loss(input_channel = opt.num_channel)
    opt.model = Model_Specification[:-3]


    print('----------------------Model Info----------------------')
    print('Root of prestrain model: ', pretrained_model_path)
    print('Model: ', opt.model)
    print('Model Specification: ', Model_Specification)
    print('Trained Date: ',opt.time)
    print('num_channel: ',opt.num_channel)
    name = input("Edit the name or press ENTER to skip: ")
    if(name!=''):
        package['name'] = name
    print('Pretrained model name: ', package['name'])
    save_model(package,pretrained_model_path)     
    model.load_state_dict(state_dict)

    print('----------------------Configure optimizer and scheduler----------------------')
    optimizer = (package['optimizer'])
    scheduler = (package['scheduler'])

    return opt,model,f_loss,optimizer,scheduler




def creating_new_model(opt):
    print('----------------------Creating model----------------------')
    opt.time = time.ctime()
    opt.epoch_ckpt = 0
    opt.val_miou = 0
    module_name = 'model.'+opt.model
    MODEL = importlib.import_module(module_name)
    model = MODEL.get_model(input_channel = opt.num_channel,is_synchoization = opt.synchonization)
    Model_Specification = MODEL.get_model_name(input_channel = opt.num_channel)
    f_loss = MODEL.get_loss(input_channel = opt.num_channel)

    print('----------------------Model Info----------------------')
    print('Root of prestrain model: ', '[No Prestrained loaded]')
    print('Model: ', opt.model)
    print('Model Specification: ', Model_Specification)
    print('Trained Date: ',opt.time)
    print('num_channel: ',opt.num_channel)
    print(opt)
    name = input("Edit the name or press ENTER to skip: ")
    #name = 'point_ring_lightDist'
    opt.model_name = name
    if(name!=''):
        opt.model_name = name
    else:
        opt.model_name = Model_Specification
    print('Model name: ', opt.model_name)
    print('----------------------Configure optimizer and scheduler----------------------')
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    experiment_dir = Path('ckpt/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(opt.model_name)
    experiment_dir.mkdir(exist_ok=True)
    shutil.copy('model/%s.py' % opt.model, str(experiment_dir))
    shutil.move(os.path.join(str(experiment_dir), '%s.py'% opt.model), 
                os.path.join(str(experiment_dir), 'model.py'))
    experiment_dir = experiment_dir.joinpath('saves')
    experiment_dir.mkdir(exist_ok=True)
    opt.save_root = str(experiment_dir)


    return opt,model,f_loss,optimizer,scheduler

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def main():
    opt = opt_global_inti()

    # This block you need to add to your main function. Tell python to open main Terminal 
    #-----------------------------------------------------------------------------------------------------------------
    opt.world_size = opt.num_gpu * opt.num_process_pgpu#world_size is total number of process you want to initiate                
    opt.num_workers = int((opt.num_workers + opt.num_process_pgpu - 1) / opt.num_process_pgpu)
    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = '66666'                      
    #------------------------------------------------------------------------------------------------------------------



    num_gpu = torch.cuda.device_count()
    assert num_gpu == opt.num_gpu,"opt.num_gpu NOT equals torch.cuda.device_count()" 

    gpu_name_list = []
    for i in range(num_gpu):
        gpu_name_list.append(torch.cuda.get_device_name(i))

    opt.gpu_list = gpu_name_list


    print('----------------------Load Dataset----------------------')
    print('Root of dataset: ', opt.dataset_root)
    print('Phase: ', opt.phase)
    print('debug: ', opt.debug)

    # I recommend to load the dataset in main function. According to torch doc. example they load in each process which i believe is not neccessary but inefficient.
    #-----------------------------------------------------------------------------------------------------------------
    train_dataset = BigredDataSet(
        root=opt.dataset_root,
        is_train=True,
        is_validation=False,
        is_test=False,
        num_channel = opt.num_channel,
        test_code = opt.debug,
        including_ring = opt.including_ring
        )
    #-----------------------------------------------------------------------------------------------------------------

    if(opt.load_pretrain!=''):
        opt,model,f_loss,_,_ = load_pretrained(opt)
    else:
        opt,model,f_loss,_,_ = creating_new_model(opt)

    f_loss.load_weight(train_dataset.labelweights)


    #I shared the memory to synchronize the weight of model on each process,just in case. you can delete this part since later DDP function will wrap it and make the copy 
    #-----------------------------------------------------------------------------------------------------------------
    model.share_memory()
    f_loss.share_memory()
    #-----------------------------------------------------------------------------------------------------------------

    #launch the multiprocess
    print('opt.world_size: ',opt.world_size)
    mp.spawn(train, nprocs=opt.world_size, args=(opt,model,f_loss,train_dataset,),join=True)




def train(rank,opt,model,f_loss,train_dataset):

    # As mp.spawn usage, the for parameter of whatever function you pass to mp.spawn usage is the process idex. I name it as rank
    gpuID= rank%opt.num_gpu

    #initialize the current process
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=opt.world_size,                              
    	rank=rank                                               
    )    

    #fix current device
    torch.cuda.set_device(gpuID)


    #wrap the model to DDP. 
    #Warning:call model.cuda(gpu) first, then make optimizer
    if(opt.apex):
        model = apex.parallel.convert_syncbn_model(model)
        model.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        model, optimizer = amp.initialize(model, optimizer,
                                        opt_level=opt.opt_level,
                                        #   keep_batchnorm_fp32=True,
                                        loss_scale="dynamic"                                      
                                        )
        model = apex.parallel.DistributedDataParallel(model,delay_allreduce=True)
    else:
        model = apex.parallel.convert_syncbn_model(model)
        model.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[gpuID])


    print('----------------------Partition Dataset and creat Dataloader----------------------')
    print('Proccess:', rank)
    print('Root of dataset: ', opt.dataset_root)
    print('Phase: ', opt.phase)
    print('debug: ', opt.debug)


    #You must partion the dataset in multiprocess, or each process train the same data which does not make any sense.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=opt.world_size,
    	rank=rank
        )

    #I decide to put same number of batches on each process. You can do whatever you want.
    #For example,suppose I want to make 12 batches in total,4 proccess,2GPU
    #I will assign 6 batches,2 process on each GPU. so Each process I load 3 batches
    #Warning:Pytorch suggest 1 process on each GPU.
    batch_num_pproc = int(opt.batch_size / opt.world_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_num_pproc,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=True,
        num_workers=opt.num_workers,
        sampler=train_sampler)

    print('train dataset num_frame: ',len(train_dataset))
    print('Batch_size: ', opt.batch_size)
    print('batch_num_pproc:',int(opt.batch_size / opt.world_size)) 
    
    print('----------------------Prepareing Training----------------------')
    metrics_list = ['Miou','Biou','Fiou','loss','OA','time_complexicity','storage_complexicity']
    manager_test = metrics_manager(metrics_list)
    metrics_list_train = ['Miou','Biou',
                            'Fiou','loss',
                            'storage_complexicity',
                            'time_complexicity']
    manager_train = metrics_manager(metrics_list_train)

    #Post result from only one process
    if(rank == 0):
        wandb.init(project=opt.wd_project,name=opt.model_name)
        wandb.config.update(opt)

    #waiting for all process
    dist.barrier()


    for epoch in range(opt.epoch_ckpt,opt.epoch_max):
        #You must call train_sampler.set_epoch on each epoch. or Next epoch will load same data.
        train_sampler.set_epoch(epoch)
        manager_train.reset()
        model.train()
        tic_epoch = time.perf_counter()
        print('---------------------Training----------------------Process: %d Epoch: %d'%(rank,epoch))
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            points, target = data
            #target.shape [B,N]
            #points.shape [B,N,C]
            points, target = points.cuda(gpuID), target.cuda(gpuID)
            #training...
            optimizer.zero_grad()
            tic = time.perf_counter()
            pred_mics = model(points)                
            toc = time.perf_counter()
            #compute loss

            #For loss
            #target.shape [B,N] ->[B*N]
            #pred.shape [B,N,2]->[B*N,2]
            loss = f_loss(pred_mics, target)            

            #pytorch will handle the weight synchonization and loss function. we dont have to do any thing
            if(opt.apex):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            #pred.shape [B,N,2] since pred returned pass F.log_softmax
            pred, target = pred_mics[0].cpu(), target.cpu()

            #pred:[B,N,2]->[B,N]
            pred = pred.data.max(dim=2)[1]
            
            #compute iou
            Biou,Fiou = mean_iou(pred,target,num_classes =2).mean(dim=0)
            miou = (Biou+Fiou)/2

            #compute Training time complexity
            time_complexity = toc - tic

            #compute Training storage complexsity
            num_device = torch.cuda.device_count()
            assert num_device == opt.num_gpu,"opt.num_gpu NOT equals torch.cuda.device_count()" 
            temp = []
            for k in range(num_device):
                temp.append(torch.cuda.memory_allocated(k))
            RAM_usagePeak = torch.tensor(temp).float().mean()
            #writeup logger
            manager_train.update('loss',loss.item())
            manager_train.update('Biou',Biou.item())
            manager_train.update('Fiou',Fiou.item())
            manager_train.update('Miou',miou.item())
            manager_train.update('time_complexicity',float(1/time_complexity))
            manager_train.update('storage_complexicity',RAM_usagePeak.item())

            # log_dict = {'loss_online':loss.item(),
            #             'Biou_online':Biou.item(),
            #             'Fiou_online':Fiou.item(),
            #             'Miou_online':miou.item(),
            #             'time_complexicity':float(1/time_complexity),
            #             'storage_complexicity_online':RAM_usagePeak.item()
            #             }
            # wandb.log(log_dict)

        dist.barrier()
        toc_epoch = time.perf_counter()
        time_tensor = torch.tensor([toc_epoch-tic_epoch]).cuda()

        #This is very important. Collect data from all process.
        #dist.all_reduce will synchonize the input return one tensor as the same shape of input.
        #dist.ReduceOp.MAX tells to return the max value for entree amongst all process
        #for details look at https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication
        dist.all_reduce(time_tensor,op=dist.ReduceOp.MAX)
        summery_dict = manager_train.summary()
        log_train_end = {}
        result_tensor = []
        

        for key in summery_dict:
            log_train_end[key+'_train_ave'] = summery_dict[key]
            result_tensor.append(summery_dict[key])
            # print(key+'_train_ave: ',summery_dict[key])
            
        result_tensor = torch.tensor(result_tensor).cuda()
        
        #synchonize all metrix. 
        dist.all_reduce(result_tensor,op=dist.ReduceOp.SUM)

        #post metrix on process 0
        if(rank == 0):
            print('after allReduce:',result_tensor)
            for i in range(len(list(log_train_end.keys()))):
                #print(i)
                key_name = list(log_train_end.keys())[i]
                if(key_name == 'Biou_train_ave' or key_name =='Miou_train_ave' 
                or key_name =='Fiou_train_ave' or key_name =='loss_train_ave'):
                    value = result_tensor[i]/opt.world_size
                else:
                    value = result_tensor[i]
                log_train_end[key_name] = value
            log_train_end['Time_PerEpoch'] = float(time_tensor.cpu()[0])
            print('log_train_end:',log_train_end)
            wandb.log(log_train_end)
        dist.barrier()
        #validation block is similar
        #for saving block. you ONLY need to save model one process. Pytorch has synchonzed model.stat_dict() for you so you dont need to do it again.
        #Rest of work still continue...


if __name__ == '__main__':
    main()
