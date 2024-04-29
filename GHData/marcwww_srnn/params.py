import torch
import torch.nn as nn
import argparse
import time
import os
import sys

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=30,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    p.add_argument('-max_length', type=int, default=50,
                   help='maximum sequence length')
    p.add_argument('-output', type=str, default='output',
                   help='output directory for model saving')
    p.add_argument('-log', type=str, default='log',
                   help='log file directory')
    p.add_argument('-hidden', type=int, default=256,
                   help='dimension of hidden states')
    p.add_argument('-stack_size', type=int, default=10,
                   help='stack size')
    p.add_argument('-stack_elem_size', type=int, default=256,
                   help='dimension of each stack element')
    p.add_argument('-nstack', type=int, default=2,
                   help='how many stacks to use')
    p.add_argument('-stack_depth', type=int, default=2,
                   help='how many stack element to use for predicting')
    p.add_argument('-gpu',type=int,default=0,
                   help='gpu index(if could be used)')
    p.add_argument('-use_stack',type=bool, default=True,
                   help='whether to use stack')
    p.add_argument('-teaching', type=float, default=0.5,
                   help='teacher forcing ratio')
    p.add_argument('-tag', type=str, default='stack',
                   help='tags to print into the log')
    p.add_argument('-mode', type=str, default='train',
                   help='train or test')
    p.add_argument('-add_pad',type=bool, default=False,
                   help='whether add one additional <PAD> for each source sentence')
    p.add_argument('-model', type=str, default='stack_ex',
                   help='stack_ex, stack or gru')
    p.add_argument('-train_file', type=str, default='aa-bb.txt',
                   help='file for training, file name format: [source name]-[target name].txt')
    p.add_argument('-test_file', type=str, default='aa-bb.test',
                   help='file for testing, file name format: [source name]-[target name].test')
    p.add_argument('-is_nl', type=bool, default=False,
                   help='whether the data is natural language')
    p.add_argument('-test_per_epoch', type=int, default=1,
                   help='every how many epoches the model test once')

    return p.parse_args()

args=parse_arguments()
device_str=args.gpu if torch.cuda.is_available() else "cpu"
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

output='./output'
log='./log'
if not os.path.exists(output):
    os.makedirs(output)
if not os.path.exists(log):
    os.makedirs(log)


name = ''.join(str(time.time()).split('.'))+'_'+'_'.join(sys.argv[1:])
enc_file = args.output + '/' + 'enc_' + name + '.pt'
dec_file = args.output + '/' + 'dec_' + name + '.pt'
log_file = args.log + '/' + 'log_' + name + '.txt'
if args.mode=='train':
    with open(log_file,'a+') as f:
        print(args,file=f)

SOS=0
EOS=1
PAD=2
NACT=3
NONLINEAR=nn.Tanh
PUSH=0
POP=1
NOOP=2
