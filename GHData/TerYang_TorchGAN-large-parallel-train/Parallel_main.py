# -*- coding: utf-8 -*-
# @Time   : 19-4-22 下午4:04
# @Author : gjj
# @contact : adau22@163.com ============================
# my github:https://github.com/TerYang/              ===
# all rights reserved                                ===
# good good study,day day up!!                       ===
# ======================================================
import argparse, os, torch,time
import numpy as np
from torch.autograd import Variable
from GAN_parallel import GAN
from CGAN import CGAN
from LSGAN_parallel import LSGAN
from DRAGAN_parallel import DRAGAN
from ACGAN import ACGAN
from WGAN_parallel import WGAN
from WGAN_GP_parallel import WGAN_GP
# from infoGAN import infoGAN
# from EBGAN import EBGAN
from BEGAN_parallel import BEGAN
import multiprocessing as mp
from readDataToGAN import *

# addr = '/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/encoding/data'#attack data

# intrusion normal dataset
# addr = '/home/yyd/dataset/intrusion/batch_scalar/ignore_ID_-1_1'#attack data
# addr = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata'#normal data
# addr = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data'#attack data

# def parse_args():
#     """parsing and configuration"""
#     desc = "Pytorch implementation of GAN collections"
#     parser = argparse.ArgumentParser(description=desc)
#
#     parser.add_argument('--gan_type', type=str, default='None',#'ACGAN',#'BEGAN',#'GAN',#'LSGAN',#default='GAN',
#                         choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
#                         help='The type of GAN')
#     # parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
#     #                     help='The name of dataset')
#     parser.add_argument('--dataset', type=str, default='None', choices=['mnist','mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
#                         help='The name of dataset')
#     parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
#     # parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
#     parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
#     parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
#     parser.add_argument('--input_size', type=int, default=21, help='The size of input image')
#     # parser.add_argument('--save_dir', type=str, default='models',
#     #                     help='Directory name to save the model')
#     parser.add_argument('--save_dir', type=str, default='models_in_cuda',
#                         help='Directory name to save the model')
#     parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
#     parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
#     parser.add_argument('--lrG', type=float, default=0.0002)
#     parser.add_argument('--lrD', type=float, default=0.0002)
#     parser.add_argument('--beta1', type=float, default=0.05)
#     parser.add_argument('--beta2', type=float, default=0.999)
#     # parser.add_argument('--gpu_mode', type=bool, default=True)
#     parser.add_argument('--gpu_mode', type=bool, default=True)
#     parser.add_argument('--benchmark_mode', type=bool, default=True)
#
#     return check_args(parser.parse_args())
#
#
# def check_args(args):
#     """checking arguments"""
#     # --save_dir
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#
#     # --result_dir
#     if not os.path.exists(args.result_dir):
#         os.makedirs(args.result_dir)
#
#     # --result_dir
#     if not os.path.exists(args.log_dir):
#         os.makedirs(args.log_dir)
#
#     # --epoch
#     try:
#         assert args.epoch >= 1
#     except:
#         print('number of epochs must be larger than or equal to one')
#
#     # --batch_size
#     try:
#         assert args.batch_size >= 1
#     except:
#         print('batch size must be larger than or equal to one')
#
#     return args
#
#
# def main():
#     """main"""
#
#     # parse arguments
#
#     args = parse_args()
#     print('Training {},started at {}'.format(args.gan_type, time.asctime(time.localtime(time.time()))))
#
#     if args is None:
#         exit()
#
#     if args.benchmark_mode:
#         torch.backends.cudnn.benchmark = True
#
#         # declare instance for GAN
#     if args.gan_type == 'GAN':
#         gan = GAN(args)
#     # elif args.gan_type == 'CGAN':
#     #     gan = CGAN(args)
#     elif args.gan_type == 'ACGAN':
#         gan = ACGAN(args)
#     # elif args.gan_type == 'infoGAN':
#     #     gan = infoGAN(args, SUPERVISED=False)
#     # elif args.gan_type == 'EBGAN':
#     #     gan = EBGAN(args)
#     elif args.gan_type == 'WGAN':
#         gan = WGAN(args)
#     elif args.gan_type == 'WGAN_GP':
#         gan = WGAN_GP(args)
#     elif args.gan_type == 'DRAGAN':
#         gan = DRAGAN(args)
#     elif args.gan_type == 'LSGAN':
#         gan = LSGAN(args)
#     elif args.gan_type == 'BEGAN':
#         gan = BEGAN(args)
#     else:
#         raise Exception("[!] There is no option for " + args.gan_type)
#
#         # launch the graph in a session
#
#     # return
#     gan.train()
#     print('Training {},finished at {}'.format(args.gan_type, time.asctime(time.localtime(time.time()))))

# normal status from hacking
addr = '/home/yyd/dataset/hacking/separateToAttackAndNormal/ignore_ID_-1_1/'#attack data


# def parse_args():
#     """parsing and configuration"""
#     desc = "Pytorch implementation of GAN collections"
#     parser = argparse.ArgumentParser(description=desc)
#
#     # parser.add_argument('--gan_type', type=str, default='None',#'ACGAN',#'BEGAN',#'GAN',#'LSGAN',#default='GAN',
#     #                     choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
#     #                     help='The type of GAN')
#     # parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
#     #                     help='The name of dataset')
#     # parser.add_argument('--dataset', type=str, default='None', choices=['mnist','mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
#     #                     help='The name of dataset')
#     # parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
#     # parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
#     parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
#     parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
#     parser.add_argument('--input_size', type=int, default=21, help='The size of input image')
#     # parser.add_argument('--save_dir', type=str, default='models',
#     #                     help='Directory name to save the model')
#     parser.add_argument('--save_dir', type=str, default='models_in_cuda',help='Directory name to save the model')
#     parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
#     parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
#     parser.add_argument('--lrG', type=float, default=0.0002)
#     parser.add_argument('--lrD', type=float, default=0.0002)
#     parser.add_argument('--beta1', type=float, default=0.05)
#     parser.add_argument('--beta2', type=float, default=0.999)
#     # parser.add_argument('--gpu_mode', type=bool, default=True)
#     parser.add_argument('--gpu_mode', type=bool, default=True)
#     parser.add_argument('--benchmark_mode', type=bool, default=True)
#
#     return check_args(parser.parse_args())
#
#
# def check_args(args):
#     """checking arguments"""
#     # --save_dir
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#
#     # --result_dir
#     if not os.path.exists(args.result_dir):
#         os.makedirs(args.result_dir)
#
#     # --result_dir
#     if not os.path.exists(args.log_dir):
#         os.makedirs(args.log_dir)
#
#     # --epoch
#     try:
#         assert args.epoch >= 1
#     except:
#         print('number of epochs must be larger than or equal to one')
#
#     # --batch_size
#     try:
#         assert args.batch_size >= 1
#     except:
#         print('batch size must be larger than or equal to one')
#
#     return args


if __name__ == '__main__':
    # main()

    # parallel run
    print('-------------------load train dataset--------------------------------------')
    data_loader = DataloadtoGAN(path=addr,mark='train',label=True,hacking=True)#single_dataset=True,label=True
    print('---------------------------------------------------------------------------\n')
    print('-------------------load validate dataset-----------------------------------')
    # valdata = DataloadtoGAN(addr,'validate')
    valdata = DataloadtoGAN(addr, mark='validate',hacking=True)
    print('---------------------------------------------------------------------------\n')

    dataset_type = 'hacking_NormalStatus_train_mixValidate'
    train_type = 'train_with_label'

    models = ['WGAN(data_loader,valdata,dataset_type,train_type).train()',
              'LSGAN(data_loader,valdata,dataset_type,train_type).train()',
              'WGAN_GP(data_loader,valdata,dataset_type,train_type).train()',
              'DRAGAN(data_loader,valdata,dataset_type,train_type).train()',
              'GAN(data_loader,valdata,dataset_type,train_type).train()',
              'BEGAN(data_loader,valdata,dataset_type,train_type).train()']
    # eval(models[3])
    for model in models:
        eval(model)

    # pool = mp.Pool(processes=len(models))
    # pool.map(eval,(models))
    # for i, model in enumerate(models):
    #     pool.apply_async(exec, (model,))
    # pool.close()
    # pool.join()

    # test = np.ones(1,10).reshape((-1,1))
    # # print(test)
    # Test_data = torch.from_numpy(test).float()

    # a = torch.zeros((10, 2)).scatter_(0, Test_data.type(torch.LongTensor), 1)
    # print(a)

    # x = torch.rand(2, 5)
    # print(x)
    # print()
    # y = torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
    # print(y)

    # class_num = 10
    # batch_size = 4
    # label = torch.from_numpy(np.array([8,9,7,6]).reshape(-1,1)).float()
    # print(label)
    # print()
    # # 然后
    # one_hot = torch.zeros(batch_size,class_num).scatter_(1, label.type(torch.LongTensor), 1)
    # print(one_hot)