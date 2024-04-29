import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse

from models import *
from utils import progress_bar

# 1.添加参数 - -local_rank
# 每个进程分配一个 local_rank 参数，表示当前进程在当前主机上的编号。
#   例如：rank=2, local_rank=0 表示第 3 个节点上的第 1 个进程。
# 这个参数是torch.distributed.launch传递过来的，我们设置位置参数来接受，local_rank代表当前程序进程使用的GPU标号
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

print(args.local_rank)

# 2.初始化使用nccl后端
# dist.init_process_group(backend='nccl')
torch.distributed.init_process_group(backend='nccl')
# When using a single GPU per process and per
# DistributedDataParallel, we need to divide the batch size
# ourselves based on the total number of GPUs we have
device_ids = [0, 1]
ngpus_per_node = len(device_ids)
batch_size = 512
batch_size = int(batch_size / ngpus_per_node)
# ps 检查nccl是否可用
# torch.distributed.is_nccl_available ()

# 3.使用DistributedSampler
# 别忘了设置pin_memory=true
# 使用 DistributedSampler 对数据集进行划分。它能帮助我们将每个 batch 划分成几个 partition，在当前进程中只需要获取和 rank 对应的那个 partition 进行训练

# train_dataset = MyDataset(train_filelist, train_labellist, args.sentence_max_size, embedding, word2id)
# train_sampler = t.utils.data.distributed.DistributedSampler(train_dataset)
# train_dataloader = DataLoader(train_dataset,
# pin_memory = true,
# shuffle = (train_sampler is None),
# batch_size = args.batch_size,
# num_workers = args.workers,
# sampler = train_sampler    )
# DataLoader：num_workers这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset,pin_memory = True,shuffle = (train_sampler is None),
                batch_size = batch_size,num_workers = 2,sampler = train_sampler)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# 注意 testset不用sampler

# 4.分布式训练
# 使用 DistributedDataParallel 包装模型，它能帮助我们为不同 GPU 上求得的梯度进行 all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）。#all reduce 后不同 GPU 中模型的梯度均为 all reduce 之前各 GPU 梯度的均值. 注意find_unused_parameters参数！

# net = textCNN(args, vectors=t.FloatTensor(wvmodel.vectors))
# if args.cuda:
# # net.cuda(device_ids[0])
#     net.cuda()
# if len(device_ids) > 1:
#     net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)


# Model
print('==> Building model..')
net = VGG('VGG19')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)  # DP
    net.cuda()
    cudnn.benchmark = True

if torch.cuda.device_count() > 1:
    net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True, device_ids=[0, 1])
    cudnn.benchmark = True
    print("0 1")


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 5.最后，把数据和模型加载到当前进程使用的GPU中，正常进行正反向传播：
# for batch_idx, (data, target) in enumerate(trainloader):
#     if device == 'cuda':
#         data, target = data.cuda(), target.cuda()
#         output = net(images)
#         loss = criterion(output, target)
#         ...
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

start_all = time.time()
for epoch in range(start_epoch, start_epoch+100):  # 200
    train(epoch)
    test(epoch)
    scheduler.step()
finish_all = time.time()

print('total_Time consumed:{:.2f}s'.format(finish_all - start_all))

# 6.在使用时，命令行调用torch.distributed.launch启动器启动：
# pytorch 为我们提供了 torch.distributed.launch 启动器，用于在命令行分布式地执行 python 文件。
# --nproc_per_node参数指定为当前主机创建的进程数。一般设定为=NUM_GPUS_YOU_HAVE当前主机的 GPU 数量，每个进程独立执行训练脚本。
# 这里是单机多卡，所以node=1，就是一台主机，一台主机上--nproc_per_node个进程
# CUDA_VISIBLE_DEVICES = 0, 1, 2, 3 python - m torch.distributed.launch - -nproc_per_node = 4 main.py
# 如果是2机3卡, nnode=2, 就是两台主机, 一台主机上--nproc_per_node=3个进程，命令应该如下（未测试过）
# python  torch.distributed.launch - -nprocs_per_node = 3 - -nnodes = 2 - -node_rank = 0
#     - -master_addr = "master-ip" - -master_port = 6005 main.py - -my
# arguments
# python
# torch.distributed.launch - -nprocs_per_node = 3 - -nnodes = 2 - -node_rank = 1 - -master_addr = "master-ip" - -master_port = 6005
# main.py - -my
# arguments

# 命令：
# CUDA_VISIBLE_DEVICES = 0, 1  python -m torch.distributed.launch - -nproc_per_node = 2 DPP.py
#
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --node_rank=1 --master_addr="master-ip" --master_port=6005  --use_env=DPP.py