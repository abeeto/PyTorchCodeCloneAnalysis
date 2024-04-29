# -*- coding: utf-8 -*-
# @Time   : 19-4-22 下午4:06
# @Author : gjj
# @contact : adau22@163.com ============================
# my github:https://github.com/TerYang/              ===
# copy from network                                  ===
# good good study,day day up!!                       ===
# ======================================================
import os, gzip, torch,argparse
import torch.nn as nn
import numpy as np
# import scipy.misc
# import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch


def parse_args():
    """parsing and configuration"""
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    # parser.add_argument('--gan_type', type=str, default='None',#'ACGAN',#'BEGAN',#'GAN',#'LSGAN',#default='GAN',
    #                     choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
    #                     help='The type of GAN')
    # parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
    #                     help='The name of dataset')

    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    # parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=21, help='The size of input image')
    # parser.add_argument('--save_dir', type=str, default='models',
    #                     help='Directory name to save the model')
    parser.add_argument('--save_dir', type=str, default='models_in_cuda',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.05)
    parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

def adjust_learning_rate(optimizer, epoch, val, lr):
    '''
    fun:Sets the learning rate to the initial LR decayed by 10 every val epochs
    :param optimizer: 优化器
    :param epoch: 当前epoch
    :param val: epoch 设置间隔
    :param lr: 初始学习率
    :return: None
    '''
    lr *= 0.1 ** (epoch // val)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def load_interval(self,epoch):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

def f2(l,r):
    import math
    if l == 0:
        if math.fabs(l - r) < 0.5:
            # TN
            return 0
        else:
            # FP
            return 3
    else:
        if math.fabs(l - r) < 0.5:
            # TP
            return 1
        else:
            # FN
            return 2

def validate(model,data_loader=None,data=None,label=None):#,mark='validate'
    '''
    validate at the same time as training
    func:select data and label or data_loader
    :return:
    '''
    import math
    f1 = lambda l,r:1 if math.fabs(l-r)<0.5 else 0
    model.eval()
    # 为测试 test mix data with normal and attack
    if data_loader is not None:
        TP = 0
        FN = 0

        TN = 0
        FP = 0
        res = {}
        count = 0
        # 默认带标签的验证集数据都传入data_loader,传入data 就可以
        for iter, (x_,l_)in enumerate(data_loader):
            # 带标签
            if iter == data_loader.dataset.__len__() // 64:
                break
            x_ = x_.cuda()
            try:
                D_real,_ = model(x_)
            except:
                D_real = model(x_)
            l_ =np.squeeze(l_.data.numpy()).tolist()
            D_real = np.squeeze(D_real.data.cpu().numpy()).tolist()
            ll = list(map(f2,l_,D_real))
            TP += ll.count(1)
            FN += ll.count(2)
            TN += ll.count(0)
            FP += ll.count(3)
            count += len(l_)
        try:
            # res['pre']='{}'.format(TP/(FP+TP))
            res['pre'] = TP / (FP + TP)
        except ZeroDivisionError:
            res['pre'] = 'NA'
        # 2  precision of negative
        try:
            res['N_pre'] = TN / (TN + FN)
            # res['N_pre']='{}'.format(TN/(TN+FN))
        except ZeroDivisionError:
            # writelog('have no P(normaly event)',file)
            res['N_pre'] = 'NA'
        # # 3 false positive rate,index of ROC , 误报 (Type I error).
        try:
            # res['FPR']='{}'.format(FP/(FP+TN))
            res['FPR'] = FP / (FP + TN)
        except ZeroDivisionError:
            res['FPR'] = 'NA'
        # 4 true positive rate,index of ROC
        try:
            # res['TPR'] ='{}'.format(TP/(TP+FN))
            res['TPR'] = TP / (TP + FN)
        except ZeroDivisionError:
            # writelog('have no P(normaly event)',file)
            res['TPR'] = 'NA'
        # 5 accurate
        try:
            # res['acc'] = (TP+NN)/len(flags)
            res['acc'] = (TP + TN) / (count)
            # results['accurate'] = accurate
        except ZeroDivisionError:
            # writelog('Error at get data,flags is None)',file)
            res['acc'] = 'NA'
        #  recall same as TPR
        try:
            res['recall'] = TP / (TP + FN)
        except ZeroDivisionError:
            # writelog('Error at get data,flags is None)',file)
            res['recall'] = 'NA'

        # F1
        try:
            res['F1'] = 2 * TP / (2 * TP + FP + FN)
        except ZeroDivisionError:
            # writelog('Error at get data,flags is None)',file)
            res['F1'] = 'NA'
        # false negative rate (Type II error).
        try:
            # res['fnr']= '{}'.format(FN/(FN+TP))
            res['fnr'] = FN / (FN + TP)
        except ZeroDivisionError:
            # writelog('Error at get data,flags is None)',file)
            res['fnr'] = 'NA'

        print('validate: D:pre:%s,N_pre:%s,acc:%s,recall:%s'%(str(res['pre']),str(res['N_pre']),str(res['acc']),str(res['recall'])),end=',')
        print('size:%d,TP:%d,TN:%d .' %(count,TP,TN),end=',')
        return res['recall']

    # 正常
    if data is not None:
        # 带标签
        # a = np.empty((3,1))
        # a.ndim
        # model = model.cuda()
        if data.__class__ == torch.Tensor:
            if data.data.numpy().ndim == 4:
                pass
            elif data.data.numpy().ndim == 3:
                data = torch.unsqueeze(data, 1)
        elif data.__class__ == np.ndarray:
            if data.numpy().ndim == 3:
                TraindataM = torch.from_numpy(data).float()  # transform to float torchTensor
                data = torch.unsqueeze(TraindataM, 1)
            elif data.numpy().ndim == 4:
                data = torch.from_numpy(data).float()  # transform to float torchTensor
        # data.cuda()
        # try:
        #     D_real, _ = model(data)
        # except:

        # cup model
        model = model.cpu()
        try:
            D_real = model(data)
        except:
            # print(data.data.cuda().numpy().shape)
            print(data.data.numpy().shape)

        if label is not None:
            # print(label.__class__,len(label))#, D_real.item(), D_real[0]

            D_real = D_real.data.numpy()
            D_real = np.squeeze(D_real).tolist()#[[],[],[]]

            ll = list(map(f1, label, D_real))
            zeros = ll.count(0)  # 错误判定
            ones = ll.count(1)  # 正确判定
            print('validate: D,size%d,errors:%d,correct:%d'%(len(ll),zeros,ones),end=',')
            print('acc:%.6f,judged as 0.'%(ones/len(ll)),end=',')
            return ones/(len(ll))
        else:
            # 验证集没有标,认为是normal 数据集,判定为0~0.5之间即可认为正确,label 0
            # D_real = D_real.data.cuda().numpy()
            D_real = D_real.data.numpy()
            D_real = np.squeeze(D_real).tolist()#[[],[],[]]

            # f = lambda x: 1 if x[0] < 0.5 else 0
            f = lambda x: 0 if x < 0.5 else 1 # test pure normal data
            ll = list(map(f, D_real))
            zeros = ll.count(0)
            ones = ll.count(1)
            print('validate: D,size:%d,zeros:%d,ones:%d'%(len(ll),zeros,ones),end=',')
            print('acc:%.6f,judged as 0' % (zeros/len(ll)), end=',')
            return zeros/len(ll)


