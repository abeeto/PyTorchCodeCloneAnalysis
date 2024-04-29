
import argparse
import json
import os
import utility
def train_options():
    parser=argparse.ArgumentParser()
    # Datasets
    parser.add_argument('--exp_name' , default='first_ex',type=str)
    parser.add_argument('--dataroot', default='./data', type=str)
    parser.add_argument('--normal_class', default='1', type=str)
    parser.add_argument('--abnormal_class', default='1', type=str)
    parser.add_argument('--datasource' , default='./data',type=str)
    parser.add_argument('--isize', default='28', type=int)
    parser.add_argument('--cropsize', default='28', type=int)
    parser.add_argument('--protocol', default ='1', type = int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--latent', default = 16, type = int)
    # Optimization options
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch', default=128, type=int, metavar='N',
                        help='batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture

        # Miscs
    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    if not os.path.isdir(args.checkpoint):
        utility.mkdir_p(args.checkpoint)

    with open(os.path.join(args.checkpoint,'configuration.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)  

    return args

def test_options():
    parser = argparse.ArgumentParser()
    # Datasets
    parser.add_argument('--exp_name' , default='first_ex',type=str)
    parser.add_argument('--dataroot', default='./data', type=str)
    parser.add_argument('--isize', default='28', type=int)
    parser.add_argument('--protocol', default ='1', type = int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    # Architecture
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args