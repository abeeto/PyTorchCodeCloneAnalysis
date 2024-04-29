#!/usr/bin/env python
# encoding: utf-8

import argparse
import numpy as np
import os
import pdb
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from PIL import Image
from os import listdir
from os.path import join
from mmd import mix_rbf_mmd2_weighted


def get_args(parser):
    parser.add_argument('--test_mix', type=str, default='', choices=['', '1090', '2080', '3070', '4060', '5050', '6040', '7030', '8020', '9010'])
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist | cifar10 | cifar100 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default='./data/mnist', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--gpu_device', type=int, default=0, help='using gpu device id')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='number of channel')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--max_iter', type=int, default=25500, help='number of generator updates to train for')
    parser.add_argument('--lambda_mmd', type=float, default=1.0, help='post-training flags, scale factor for MMD, exclusively in errG')
    parser.add_argument('--lambda_ae', type=float, default=8.0, help='post-training flags, scale factor for autoencoder')
    parser.add_argument('--lambda_rg', type=float, default=16.0, help='post-training flags, scale factor for hinge')
    parser.add_argument('--glr', type=float, default=0.00005, help='learning rate, default=0.00005')
    parser.add_argument('--dlr', type=float, default=0.00005, help='learning rate, default=0.00005')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--d_calibration_step', type=int, default=500, help='number of G iters between big n=100 Diter steps')
    parser.add_argument('--schedule', type=str, default='constant', help='schedule of Diters per Giter', choices=['original', 'constant'])
    parser.add_argument('--exp_const', type=float, default=0.05, help='constant from thinning_kernel_part')
    parser.add_argument('--thin_type', type=str, default='logistic', help='type of thinning function', choices=['kernel', 'logistic'])
    parser.add_argument('--thinning_scale', type=float, default=0.75, help='Maximum of thinning_kernel. 1 / (1 - thinning_scale) defines max weight')
    parser.add_argument('--load_existing', type=int, default=0, help='Reference number of saved state file')
    parser.add_argument('--num_pretrain', type=int, default=-1, help='Number of pretrain runs before weighted MMD')
    parser.add_argument('--tag', type=str, default='test', help='tag pre-pended to save_dir')
    parser.add_argument('--diagnostic', type=int, default=0, choices=[0, 1])
    return parser


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class FolderWithImages(data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        super(FolderWithImages, self).__init__()
        self.image_filenames = [join(root, x)
                                for x in listdir(root) if is_image_file(x.lower())]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class ALICropAndScale(object):
    def __call__(self, img):
        return img.resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))


def get_data(args, train_flag=True, mix=None):
    transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transform)

    elif args.dataset == 'lsun':
        dataset = dset.LSUN(db_path=args.dataroot,
                            classes=['bedroom_train'],
                            transform=transform)

    elif args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=args.dataroot,
                               download=True,
                               train=train_flag,
                               transform=transform)

    elif args.dataset == 'cifar100':
        dataset = dset.CIFAR100(root=args.dataroot,
                                download=True,
                                train=train_flag,
                                transform=transform)

    elif args.dataset == 'mnist':
        dataset = dset.MNIST(root=args.dataroot,
                             download=True,
                             train=train_flag,
                             transform=transform)
        # Fetch only zeros and twos.
        dataset_main = (
            [v for i,v in enumerate(dataset) if dataset.train_labels[i] == 0])
        dataset_target = (
            [v for i,v in enumerate(dataset) if dataset.train_labels[i] == 1])
        minlen = min(len(dataset_main), len(dataset_target))
        dataset_main = dataset_main[:minlen]
        dataset_target = dataset_target[:minlen]
        if mix is None:
            dataset_mixed_8020 = dataset_main + dataset_target[: minlen / 4]
            dataset_mixed_5050 = dataset_main + dataset_target

            print(('Created set of size {}/{} = {}, '
                'and set of size {}/{} = {}').format(
                    minlen, len(dataset_target[: minlen / 4]),
                    float(minlen / len(dataset_target[: minlen / 4])), minlen,
                    minlen, minlen / minlen))

        # For user-defined proportion, return only the prescribed mix.
        if mix is not None:
            p0 = int(mix[:2])  # E.g. gets 20 from '2080'
            p1 = int(mix[2:])  # E.g. gets 80 from '2080'
            n0 = int(p0 / 100. * minlen)
            n1 = minlen - n0
            print('Mix. Created set of size {}/{}'.format(n0, n1))
            dataset_main = dataset_main[:n0] 
            dataset_target = dataset_target[:n1]
            prescribed_mix = dataset_main + dataset_target
            return prescribed_mix, None, dataset_main, dataset_target 

        return (dataset_mixed_8020, dataset_mixed_5050, dataset_main,
            dataset_target)

    elif args.dataset == 'celeba':
        imdir = 'train' if train_flag else 'val'
        dataroot = os.path.join(args.dataroot, imdir)
        if args.image_size != 64:
            raise ValueError('the image size for CelebA dataset need to be 64!')

        dataset = FolderWithImages(root=dataroot,
                                   input_transform=transforms.Compose([
                                       ALICropAndScale(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]),
                                   target_transform=transforms.ToTensor()
                                   )
    else:
        raise ValueError("Unknown dataset %s" % (args.dataset))
    return dataset


def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'


def test_wmmd():
    # Get args.
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    # Config settings for wmmd test.
    wmmd_per_mix = []
    num_tests_per_mix = 10
    mixes = ['1090', '2080', '3070', '4060', '5050', '6040', '7030', '8020',
        '9010']
    # Get base data, the point of comparison for all other mixes.
    trn_dataset_base = get_data(args, train_flag=True, mix='8020')
    trn_loader_base = torch.utils.data.DataLoader(trn_dataset_base,
        batch_size=10, shuffle=True, num_workers=int(args.workers))
    for mix in mixes:
        # Prepare loader for a particular mix.
        trn_dataset_mix = get_data(args, train_flag=True, mix=mix)
        trn_loader_mix = torch.utils.data.DataLoader(trn_dataset_mix,
            batch_size=10, shuffle=True, num_workers=int(args.workers))

        wmmd_samples = []
        for _ in range(num_tests_per_mix):
            # Get batches of base and mix.
            batch_base, labels_base = iter(trn_loader_base).next()
            batch_mix, labels_mix = iter(trn_loader_mix).next()
            scaled_labels_base = (
                args.thinning_scale * labels_base.type(torch.FloatTensor))
            # Compute wmmd2 between samples.
            enc_base, _ = netD(Variable(batch_base.cuda())) 
            enc_mix, _ = netD(Variable(batch_mix.cuda())) 
            sigma_list = [1., 2., 4., 8., 16.]
            wmmd2 = mix_rbf_mmd2_weighted(enc_base, enc_mix, sigma_list,
                args.exp_constant, args.thinning_scale,
                x_enc_p1=scaled_labels_base)
            wmmd_samples.append(wmmd2)

