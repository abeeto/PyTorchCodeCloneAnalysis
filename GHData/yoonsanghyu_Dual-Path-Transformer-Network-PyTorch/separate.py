#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU
from collections import OrderedDict
import argparse
import os

import librosa
import soundfile as sf
import torch

from data import EvalDataLoader, EvalDataset
from dptnet import DPTNet
from utils import remove_pad


parser = argparse.ArgumentParser('Separate speech using DPTNet')

# Network architecture
parser.add_argument('--N', default=64, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--C', default=2, type=int, 
                    help='Maximum number of speakers')
parser.add_argument('--L', default=4, type=int, 
                    help='Length of window in autoencoder') # L=2 in paper
parser.add_argument('--H', default=4, type=int, 
                    help='Number of head in Multi-head attention')
parser.add_argument('--K', default=250, type=int, 
                    help='segment size')
parser.add_argument('--B', default=6, type=int, 
                    help='Number of repeats')


parser.add_argument('--model_path', type=str, default='exp/temp/temp_best.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture wav files')
parser.add_argument('--mix_json', type=str, default='data/tt/mix.json',
                    help='Json file including mixture wav files')
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    model = DPTNet(args.N, args.C, args.L, args.H, args.K, args.B)
    
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model_info = torch.load(args.model_path)

    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)
    
    
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_dir, args.mix_json,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate)
    eval_loader =  EvalDataLoader(eval_dataset, batch_size=1)
    os.makedirs(args.out_dir, exist_ok=True)

    def write(inputs, filename, sr=args.sample_rate):
        sf.write(filename, inputs, sr)
        #librosa.output.write_wav(filename, inputs, sr)# norm=True)

    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames = data
            if args.use_cuda:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()
            # Forward
            estimate_source = model(mixture)  # [B, C, T]
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            mixture = remove_pad(mixture, mix_lengths)
            # Write result
            for i, filename in enumerate(filenames):
                filename = os.path.join(args.out_dir,
                                        os.path.basename(filename).strip('.wav'))
                write(mixture[i], filename + '.wav')
                C = flat_estimate[i].shape[0]
                for c in range(C):
                    write(flat_estimate[i][c], filename + '_s{}.wav'.format(c+1))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)

