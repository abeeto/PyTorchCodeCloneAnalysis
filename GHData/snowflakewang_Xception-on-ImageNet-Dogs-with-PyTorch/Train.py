import argparse#argparse是python自带的命令行参数解析包，用来方便地读取命令行参数
import builtins
import math
import os
import random
import shutil
import time
import warnings
import collections
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import Xception

parser=argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir',metavar='DIR',help='path to dataset')
parser.add_argument('-a','--arch',metavar='ARCH',default='Xception',choices=['Xception'],help='model architecture: XceptionNet')
parser.add_argument('-j','--workers',default=1,type=int,metavar='N',help='number of data loading workers(default: 1)')
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number(useful on restars)')
parser.add_argument('-b','--batch_size',default=256,type=int,metavar='N',help='mini-batch size(default: 256), this is the total batch size of all GPUs on the current node'
                                                                              'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr','--learning rate',default=0.03,type=float,metavar='LR',help='initial learning rate',dest='lr')
parser.add_argument('--schedule',default=[120,160],nargs='*',type=int,help='learning rate schedule(when to drop lr by 10x)')
parser.add_argument('--momentum',default=0.9,type=float,metavar='M',help='momentum of SGD solver')
parser.add_argument('--wd','--weight_decay',default=1e-4,type=float,metavar='W',help='weight decay(default: 1e-4)')
parser.add_argument('-p','--print_freq',default=10,type=int,metavar='N',help='print frequency(default: 10)')
parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to latest checkpoint(default: None)')
parser.add_argument('--world_size',default=-1,type=int,help='number of nodes for distributed training')
parser.add_argument('--rank',default=-1,type=int,help='node rank for distributed training')
parser.add_argument('--dist_url',default=None,type=str,help='url used to set up distributed training')
parser.add_argument('--dist_backend',default='nccl',type=str,help='distributed backend')
parser.add_argument('--seed',default=None,type=int,help='seed for initializing training')
parser.add_argument('--gpu',default=None,type=int,help='GPU id to use')
parser.add_argument('--multiprocessing_distributed',default=False,action='store_true',help='use multi-processing distributed training to launch'
                                                                             'N processes per node, which has N GPUs.'
                                                                             'This is the fastest way to use PyTorch for either single node'
                                                                          'or multi node data parallel training')
#xception specific configs
parser.add_argument('--pretrained',default=False,type=bool,help='pretrain the network(default: False)')

#metavar意为在使用方法消息中使用的参数值实例
#dest意为被添加到parse_args()所返回对象上的属性名

def main():
    args = parser.parse_args()#ArgumentParser的实例parser通过parse_args()方法来解析参数

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)#为cpu设置种子，生成随机数
        cudnn.deterministic = True#cudnn.benchmark对于输入数据维度和类型变化不大、网络结构较为固定的模型有加速作用，cudnn.benchmark设置为true后cuDNN会自动寻找最适合当前配置的高效算法
        #benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异，想避免这种波动，就设置deterministic为true
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:#是否装备有GPU，即后续是否要使用GPU进行训练
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:#是否要采用采用分布式计算技术
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def read_csv_labels(fname):
    '''读取文件，返回图片名称到标签之间的映射'''
    with open(fname,'r') as f:
        lines=f.readlines()[1:]#跳过表头
        #此处的readlines()函数需要注意，python中有read()，readline()，readlines()三种函数
        #read()
        # 语法：fileObject.read([size])
        # fileObject：打开的文件对象
        # size：可选参数，用于指定一次最多可读取的字符（字节）个数，如果省略，则默认一次性读取所有内容。
        # read()方法用于逐个字节（或者逐个字符）读取文件中的内容，需要借助open() 函数，并以可读模式（包括 r、r+、rb、rb+）打开文件。

        # readline()
        # 语法：fileObject.readline([size])
        # fileObject：打开的文件对象
        # size：可选参数，用于指定读取每一行时，一次最多读取的字节数。
        # readline() 方法用于从文件读取整行，包括 "\n" 字符。readline()读取文件数据的前提是使用open() 函数指定打开文件的模式必须为可读模式（包括 r、rb、r+、rb+）

        # readlines()
        # 语法：fileObject.readlines()
        # fileObject：打开的文件对象
        # readlines() 方法用于一次性读取所有行并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
    tokens=[l.rstrip().split(',') for l in lines]
    #python中有三种去除头尾字符，空白符的函数
    #strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    #lstrip：用来去除开头字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    #rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    #string.strip([chars])
    #string.lstrip([chars])
    #string.rstrip([chars])
    #参数chars是可选的，当chars为空，默认删除string头尾的空白符(包括\n、\r、\t、' ')
    #当chars不为空时，函数会被chars解成一个个的字符，然后将这些字符去掉。
    #它返回的是去除头尾字符(或空白符)的string副本，string本身不会发生改变。
    return dict(((name,label) for name,label in tokens))

def copyfile(filename,target_dir):
    '''将文件复制到目标路径下'''
    os.makedirs(target_dir,exist_ok=True)#在路径不存在的情况下创建路径，os.makedirs()用来创造多层目录（与此对应地，os.mkdir()只创造单层目录，如果os.mkdir()要创造的目录的之前的根目录有一些是不存在的，就会报错）
    #os.makedirs(name, mode=0o777, exist_ok=False)，name：你想创建的目录名，mode：要为目录设置的权限数字模式，默认的模式为 0o777 (八进制)，
    #exist_ok：是否在目录存在时触发异常。如果exist_ok为False（默认值），则在目标目录已存在的情况下触发FileExistsError异常；如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常
    shutil.copy(filename,target_dir)
    #关于shutil模块，该模块是用来进行文件操作的
    #shutil.copyfile(src,dst)：
    #src是需要被操作的文件名，dst是该文件需要被复制到的地方，注意dst应该是路径加文件名而不是单纯的路径，举个例子：copyfile(data_dir+'xxx.jpg',target_dir+'xxx.jpg')而不能写成copyfile(data_dir+'xxx.jpg',target_dir)
    #shutil.copy(src,dst)：
    #注意这与copyfile()略有不同，src是需要被操作的文件名，dst是文件名或者路径名

def reorg_train_valid(data_dir,labels,valid_ratio):
    n=collections.Counter(labels.values()).most_common()[-1][1]#训练集中样本数最少的类别包含的样本数，
    #其中Counter()统计“可迭代序列”中每个元素的出现的次数，Counter().most_common(n)统计出现次数最多的前n个元素，当n未给定时，列出全部元素及其出现次数
    #本句中，对most_common()数组取最后一个元素（一个二元元组），即得样本数最少的类别，再取元组中的后一个元素，即得到样本数最少的类别包含的样本数
    n_valid_per_label=max(1,math.floor(n*valid_ratio))#验证集中每一类的样本数，其中math.floor(n)函数给出不大于n的最大整数
    #n*valid_ratio代表训练集样本最少的一类可以被作为验证集样本的数量，如果超过了这个数，就会有至少某一类图片在验证集中数量不足的情况
    label_count={}#创建一个字典，其索引用于表征train_valid_test/train_valid/路径下已存在的以label命名的文件夹，其内容用于表征label文件夹中文件数量
    for train_file in os.listdir(os.path.join(data_dir,'train')):
        label=labels[train_file.split('.')[0]]#为每个训练集样本匹配类别，其中.split()通过指定分隔符对字符串进行切片
        #train_file是一个文件名字符串，如xxxx.jpg，因此train_file.split('.')将文件名字符串的后缀与前缀分开，第一元是文件名，第二元是jpg，[0]代表取第一元即文件名，该文件名作为字典labels的索引，给label赋予该文件对应的标签
        fname=os.path.join(data_dir,'train',train_file)#获得train_file文件的路径
        copyfile(fname,os.path.join(data_dir,'train_valid_test','train_valid',label))#将train_file文件保存到train_valid_test/train_valid/路径下的以其label命名的文件夹中
        if label not in label_count or label_count[label]<n_valid_per_label:#如果验证集中还没有这种label或者这种label的验证集样本数还不足
            copyfile(fname,os.path.join(data_dir,'train_valid_test','valid',label))
            label_count[label]=label_count.get(label,0)+1#当label_count字典中能够查询到label索引时，返回label索引对应的值，这对应label_count[label]<n_valid_per_label情况，说明label类别已存在但还未达到所需数量；
            #若不能查询到，返回括号中后面的值，即0，这对应label not in label_count情况，说明label类别还不存在需要先创建，然后label_count.get(label,0)+1=0+1=1意为现在label类别中有一个文件
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))#构造的验证集达到了类别(label not in label_count)和每个类别中数量(label_count[label]<n_valid_per_label)的要求后，剩下的文件用于构造训练集
    return n_valid_per_label

def reorg_test(data_dir):#用该函数整理测试集，从而方便预测时读取
    for test_file in os.listdir(os.path.join(data_dir,'test')):
        copyfile(os.path.join(data_dir,'test',test_file),os.path.join(data_dir,'train_valid_test','test','unknown'))
        #测试集数据整理，其实就是把data_dir/test中所有文件拷贝到data_dir/train_valid_test/test/unknown中，类别只有一类，名为unknown

def reorg_dog_data(data_dir,valid_ratio):
    labels=read_csv_labels(os.path.join(data_dir,'labels.csv'))#获得字典labels，索引是图片文件名称，键值是类别
    reorg_train_valid(data_dir,labels,valid_ratio)#将原训练集train（或者说train_valid，因为对原训练集train分类后得到train_valid）分割成新的训练集train和验证集valid并与标签集labels匹配
    reorg_test(data_dir)#整理出测试集test，里面包含仅一类成为unknown

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model=Xception.xception(args.pretrained)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        pass
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        #raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    batch_size,valid_ratio = 8,0.1
    #reorg_dog_data(args.data_dir,valid_ratio)

    transform_train = torchvision.transforms.Compose(
        [torchvision.transforms.RandomResizedCrop(299, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize(384),
                                                     torchvision.transforms.CenterCrop(299),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])

    train_ds, train_valid_ds = [
        torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'train_valid_test', folder), transform=transform_train)
        for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [
        torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'train_valid_test', folder), transform=transform_test)
        for folder in ['valid', 'test']]
    # 关于这里使用的ImageFolder，它是一个通用的数据加载器，它要求以data_dir/xxx.jpg（或者是png等其他格式）来组织数据集的训练、验证和测试
    # 先来看看torchvision.datasets.ImageFolder(root,transform,target_transform,loader)，root代表图片存储的根目录，即各类别文件夹所在目录的上一层目录；transform是对图片进行的预处理操作（函数），
    # 原始图片作为输入，返回转换后的图片；target_transform是对图片类别进行的预处理操作，输入为 target，输出对其的转换。如果不传该参数，即对 target 不做任何转换，返回的顺序索引 0,1, 2…
    # 该函数的返回值有三种属性，
    # 为self.classes：用一个 list 保存类别名称
    # self.class_to_idx：类别对应的索引，与不做任何转换返回的 target 对应
    # self.imgs：保存(img_path, class) tuple的 list，其中img_path指的是文件的路径

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for
                                    dataset in (train_ds, train_valid_ds)]
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

    '''
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    '''

    for epoch in range(args.start_epoch, args.epochs):

        '''
        if args.distributed:
            train_sampler.set_epoch(epoch)
        '''

        '''
        adjust_learning_rate(optimizer, epoch, args)
        '''

        # train for one epoch
        train(train_iter,test_iter, model, criterion, optimizer, epoch, args)

        '''
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
        '''

class AverageMeter(object):
    '''computes and stores the average and current value'''
    def __init__(self,name,format=':f'):
        self.name=name
        self.format=format
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

    def __str__(self):
        format_str='{name} {val'+self.format+'} ({avg'+self.format+'})'
        return format_str.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self,num_batches,meters,prefix=''):
        self.batch_format_str=self._get_batch_format_str(num_batches)
        self.meters=meters
        self.prefix=prefix

    def display(self,batch):
        entries=[self.prefix+self.batch_format_str.format(batch)]
        entries+=[str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_format_str(self,num_batches):
        num_digits=len(str(num_batches//1))
        format='{:'+str(num_digits)+'d}'
        return '['+format+'/'+format.format(num_batches)+']'

def evaluate_loss(data_iter,net,criterion,args):
    l_sum,n=0.0,0
    for features,labels in data_iter:
        features,labels=features.to(args.gpu),labels.to(args.gpu)
        output=net(features)
        l=criterion(output,labels)
        l_sum=l.sum()
        n+=labels.numel()
    return  l_sum/n

#计算准确率的函数
def accuracy(data_iter, net, args):
    if args.gpu is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    elif isinstance(net,torch.nn.Module):
        device=args.gpu
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum

def train(train_iter,test_iter,model,criterion,optimizer,epoch,args):
    batch_time=AverageMeter('Time',':6.3f')
    data_time=AverageMeter('Data',':6.3f')
    losses=AverageMeter('Loss',':.4e')
    top1=AverageMeter('Acc@1',':6.2f')
    top5=AverageMeter('Acc@5',':6.2f')
    progress=ProgressMeter(len(train_iter),[batch_time,data_time,criterion,top1,top5],prefix='Epoch: [{}]'.format(epoch))

    model.train()#调换到训练模式
    if args.gpu is not None:
        model=model.to(args.gpu)
    batch_count=0
    end=time.time()
    for i,(X,y) in enumerate(train_iter):
        data_time.update(time.time()-end)
        train_l_sum, train_acc_sum, n,start= 0.0, 0.0, 0,time.time()

        for X,y in train_iter:
            if args.gpu is not None:
                X=X.to(args.gpu)
                y=y.to(args.gpu)

            output=model(X)
            loss=criterion(output,y)

            losses.update(loss.item(),X.shape[0])

            optimizer.zero_grad()#已经累计的梯度清零
            loss.backward()#反向传播算出各参数的梯度
            optimizer.step()#进行一次权重更新
            train_l_sum += loss.cpu().item()
            train_acc_sum += (output.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            print('the batch_count is',batch_count)
        acc1=train_acc_sum / n
        top1.update(acc1, X.shape[0])
        # top5.update(acc5[0],X[0].shape[0])
        test_acc = accuracy(test_iter, model)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, acc1, test_acc, time.time() - start))

        batch_time.update(time.time()-end)

        if i%args.print_freq==0:
            progress.display(i)

if __name__=='__main__':
    main()