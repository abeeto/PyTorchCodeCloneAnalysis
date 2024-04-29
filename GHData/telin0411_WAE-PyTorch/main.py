"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
import argparse
import os
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data


def get_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['celebA'], help='celebA')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector, noise')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--print_every', type=int, default=10, help='number iterations to print out statements')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--e_pretrain', action='store_true', help='if pretrain encoder')
    parser.add_argument('--e_pretrain_sample_size', type=int, default=256, help='sample size for encoder pretrain')
    parser.add_argument('--e_pretrain_iters', type=int, default=1, help='max epochs to pretrain the encoder')
    parser.add_argument('--input_normalize_sym', action='store_true', help='for tanh of GAN outputs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--checkpoint', default='', help="path to checkpoint (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--noise', default='gaussian', choices=['gaussian', 'add_noise'], help='noise type for WAE, | gaussian | add_noise |')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
    parser.add_argument('--LAMBDA', type=float, default=100, help='LAMBDA for WAE')
    parser.add_argument('--img_norm', type=float, default=None, help='normalization of images')
    parser.add_argument('--mode', type=str, default='gan', choices=['gan', 'mmd'], help='| gan | mmd |')
    parser.add_argument('--kernel', type=str, default='IMQ', choices=['RBF', 'IMQ'], help='| RBF | IMQ |')
    parser.add_argument('--pz_scale', type=float, default=1., help='sacling of sample noise')
    opt = parser.parse_args()
    print(opt)
    return opt


def main():
    opt = get_parsers()
    # specify the gpu id if using only 1 gpu
    if opt.ngpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    # output directory
    os.makedirs(opt.outf, exist_ok=True)

    # random seeds
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)

    # use cuda
    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # main training
    if opt.mode == 'gan':
        from train_wae_gan import train
    elif opt.mode == 'mmd':
        from train_wae_mmd import train
    else:
        raise NotImplementedError
    train(opt)


if __name__ == "__main__":
    main()
