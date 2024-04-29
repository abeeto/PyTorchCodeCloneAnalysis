#!/usr/bin/env python
import torch
import os
from glob import glob
from argparse import ArgumentParser
from general_utils import print_progress
from collections import OrderedDict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('nets_folder', type=str, help='path to trained networks folder')
    parser.add_argument('-p', '--parameters', nargs='+', default=None)
    args = parser.parse_args()
    device = torch.device('cpu')
    print_out = {}
    folders = glob(os.path.join(args.nets_folder, '*'))
    for ind, net_path in enumerate(folders):
        sep = (70 - len(net_path)) * ' '
        out_str = [net_path + sep]
        mtime = [os.path.getmtime(net_path)]
        for cp_name, cp_type in zip(['last_checkpoint.pth', 'checkpoint.pth'], ['last', 'best']):
            cp_path = os.path.join(net_path, 'checkpoints', cp_name)
            if os.path.exists(cp_path):
                try:
                    checkpoint = torch.load(cp_path, map_location=device)
                    epoch = checkpoint['epoch']
                    out_str.append('{}: {}\t'.format(cp_type, epoch))
                    mtime.append(os.path.getmtime(cp_path))
                except RuntimeError:
                    print('failed to load', cp_path)
            else:
                out_str.append('{}:    \t'.format(cp_type))
        if args.parameters is not None:
            prm_path = os.path.join(net_path, 'cfg.yaml')
            with open(prm_path) as f:
                lines = f.readlines()
            for line in lines:
                if any(p in line for p in args.parameters):
                    out_str.append(line.strip() + '\t')
        print_progress(iteration=ind+1, total=len(folders), prefix='-->',
                       suffix='reading {} \t\t\t'.format(os.path.basename(net_path)), length=30)
        print_out[max(mtime)] = out_str
    print()
    od = OrderedDict(sorted(print_out.items()))
    for v in od.values():
        print('| '.join(v))

