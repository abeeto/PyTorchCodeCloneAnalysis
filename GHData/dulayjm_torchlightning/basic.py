from __future__ import absolute_import

import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

from torch.utils.data import DataLoader, random_split
import os
from pytorch_metric_learning import losses, samplers
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset
import optuna
import gc

import pytorch_lightning as pl
from argparse import ArgumentParser

# DF-Net Imports
from collections import defaultdict
import cv2
# import matplotlib.pyplot as plt
from itertools import islice
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
import torch.nn.functional as F
import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import random
from skimage import io, transform
from sklearn.manifold import TSNE
import time
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler


# DF-Net Utils
def resize_like(x, target, mode='bilinear'):
    return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)


def list2nparray(lst, dtype=None):
    """fast conversion from nested list to ndarray by pre-allocating space"""
    if isinstance(lst, np.ndarray):
        return lst
    assert isinstance(lst, (list, tuple)), 'bad type: {}'.format(type(lst))
    assert lst, 'attempt to convert empty list to np array'
    if isinstance(lst[0], np.ndarray):
        dim1 = lst[0].shape
        assert all(i.shape == dim1 for i in lst)
        if dtype is None:
            dtype = lst[0].dtype
            assert all(i.dtype == dtype for i in lst), 'bad dtype: {} {}'.format(dtype, set(i.dtype for i in lst))
    elif isinstance(lst[0], (int, float, complex, np.number)):
        return np.array(lst, dtype=dtype)
    else:
        dim1 = list2nparray(lst[0])
        if dtype is None:
            dtype = dim1.dtype
        dim1 = dim1.shape
    shape = [len(lst)] + list(dim1)
    rst = np.empty(shape, dtype=dtype)
    for idx, i in enumerate(lst):
        rst[idx] = i
    return rst


def get_img_list(path):
    return sorted(list(Path(path).glob('*.png'))) + sorted(list(Path(path).glob('*.jpg'))) + sorted(
        list(Path(path).glob('*.jpeg')))


def gen_miss(img, mask, output):
    imgs = get_img_list(img)
    masks = get_img_list(mask)
    print('Total images:', len(imgs), len(masks))

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for i, (img, mask) in tqdm.tqdm(enumerate(zip(imgs, masks))):
        path = out.joinpath('miss_%04d.png' % (i + 1))
        img = cv2.imread(str(img), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img.shape[:2][::-1])
        mask = mask[..., np.newaxis]
        miss = img * (mask > 127) + 255 * (mask <= 127)

        cv2.imwrite(str(path), miss)


def merge_imgs(dirs, output, row=1, gap=2, res=512):
    image_list = [get_img_list(path) for path in dirs]
    img_count = [len(image) for image in image_list]
    print('Total images:', img_count)
    assert min(img_count) > 0, 'Please check the path of empty folder.'

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_img = len(dirs)
    row = row
    column = (n_img - 1) // row + 1
    print('Row:', row)
    print('Column:', column)

    for i, unit in tqdm.tqdm(enumerate(zip(*image_list))):
        name = output_dir.joinpath('merge_%04d.png' % i)
        merge = np.ones([
            res * row + (row + 1) * gap, res * column + (column + 1) * gap, 3], np.uint8) * 255
        for j, img in enumerate(unit):
            r = j // column
            c = j - r * column
            img = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if img.shape[:2] != (res, res):
                img = cv2.resize(img, (res, res))
            start_h, start_w = (r + 1) * gap + r * res, (c + 1) * gap + c * res
            merge[start_h: start_h + res, start_w: start_w + res] = img
        cv2.imwrite(str(name), merge)


# DF-Net Model
def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_activation(name):
    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'elu':
        activation == nn.ELU()
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = self.conv_same_pad(kernel_size, stride)
        if type(padding) is not tuple:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding * 2, 0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
            )

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding)

    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def forward(self, x):
        return self.trans_conv(x)


class UpBlock(nn.Module):

    def __init__(self, mode='nearest', scale=2, channel=None, kernel_size=4):
        super().__init__()

        self.mode = mode
        if mode == 'deconv':
            self.up = ConvTranspose2dSame(
                channel, channel, kernel_size, stride=scale)
        else:
            def upsample(x):
                return F.interpolate(x, scale_factor=scale, mode=mode)

            self.up = upsample

    def forward(self, x):
        return self.up(x)


class EncodeBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            normalization=None, activation=None):
        super().__init__()

        self.c_in = in_channels
        self.c_out = out_channels

        layers = []
        layers.append(
            Conv2dSame(self.c_in, self.c_out, kernel_size, stride))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecodeBlock(nn.Module):

    def __init__(
            self, c_from_up, c_from_down, c_out, mode='nearest',
            kernel_size=4, scale=2, normalization='batch', activation='relu'):
        super().__init__()

        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpBlock(mode, scale, c_from_up, kernel_size=scale)

        layers = []
        layers.append(
            Conv2dSame(self.c_in, self.c_out, kernel_size, stride=1))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.decode = nn.Sequential(*layers)

    def forward(self, x, concat=None):
        out = self.up(x)
        if self.c_from_down > 0:
            out = torch.cat([out, concat], dim=1)
        out = self.decode(out)
        return out


class BlendBlock(nn.Module):

    def __init__(
            self, c_in, c_out, ksize_mid=3, norm='batch', act='leaky_relu'):
        super().__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = nn.Sequential(
            Conv2dSame(c_in, c_mid, 1, 1),
            get_norm(norm, c_mid),
            get_activation(act),
            Conv2dSame(c_mid, c_out, ksize_mid, 1),
            get_norm(norm, c_out),
            get_activation(act),
            Conv2dSame(c_out, c_out, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.blend(x)


class FusionBlock(nn.Module):
    def __init__(self, c_feat, c_alpha=1):
        super().__init__()
        c_img = 3
        self.map2img = nn.Sequential(
            Conv2dSame(c_feat, c_img, 1, 1),
            nn.Sigmoid())
        self.blend = BlendBlock(c_img * 2, c_alpha)

    def forward(self, img_miss, feat_de):
        img_miss = resize_like(img_miss, feat_de)
        raw = self.map2img(feat_de)
        alpha = self.blend(torch.cat([img_miss, raw], dim=1))
        result = alpha * raw + (1 - alpha) * img_miss
        return result, alpha, raw


class DFNet(nn.Module):
    def __init__(
            self, c_img=3, c_mask=1, c_alpha=3,
            mode='nearest', norm='batch', act_en='relu', act_de='leaky_relu',
            en_ksize=[7, 5, 5, 3, 3, 3, 3, 3], de_ksize=[3] * 8,
            blend_layers=[0, 1, 2, 3, 4, 5]):
        super().__init__()

        c_init = c_img + c_mask

        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        assert self.n_en == self.n_de, (
            'The number layer of Encoder and Decoder must be equal.')
        assert self.n_en >= 1, (
            'The number layer of Encoder and Decoder must be greater than 1.')

        assert 0 in blend_layers, 'Layer 0 must be blended.'

        self.en = []
        c_in = c_init
        self.en.append(
            EncodeBlock(c_in, 64, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            c_in = self.en[-1].c_out
            c_out = min(c_in * 2, 512)
            self.en.append(EncodeBlock(
                c_in, c_out, k_en, stride=2,
                normalization=norm, activation=act_en))

        # register parameters
        for i, en in enumerate(self.en):
            self.__setattr__('en_{}'.format(i), en)

        self.de = []
        self.fuse = []
        for i, k_de in enumerate(de_ksize):

            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i - 1].c_in
            layer_idx = self.n_de - i - 1

            self.de.append(DecodeBlock(
                c_from_up, c_from_down, c_out, mode, k_de, scale=2,
                normalization=norm, activation=act_de))
            if layer_idx in blend_layers:
                self.fuse.append(FusionBlock(c_out, c_alpha))
            else:
                self.fuse.append(None)

        # register parameters
        for i, de in enumerate(self.de[::-1]):
            self.__setattr__('de_{}'.format(i), de)
        for i, fuse in enumerate(self.fuse[::-1]):
            if fuse:
                self.__setattr__('fuse_{}'.format(i), fuse)

    def forward(self, img_miss, mask):
        out = torch.cat([img_miss, mask], dim=1)

        out_en = [out]
        for encode in self.en:
            out = encode(out)
            out_en.append(out)

        results = []
        alphas = []
        raws = []
        for i, (decode, fuse) in enumerate(zip(self.de, self.fuse)):
            out = decode(out, out_en[-i - 2])
            if fuse:
                result, alpha, raw = fuse(img_miss, out)
                results.append(result)
                alphas.append(alpha)
                raws.append(raw)

        return results[::-1], alphas[::-1], raws[::-1]


# Inpainter
class Inpainter:
    def __init__(self, model_path, input_size, batch_size):
        self.model_path = model_path
        self._input_size = input_size
        self.batch_size = batch_size
        self.init_model(model_path)


    @property
    def input_size(self):
        if self._input_size > 0:
            return (self._input_size, self._input_size)
        elif 'celeba' in self.model_path:
            return (256, 256)
        else:
            return (256, 256)


    def init_model(self, path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')
            print('Using gpu.')
        else:
            self.device = torch.device('cpu')
            print('Using cpu.')

        self.model = DFNet().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint) # loading in that pre-trained model
        self.model.eval()

        print('Model %s loaded.' % path)


    def to_numpy(self, tensor):
        tensor = tensor.mul(255).byte().data.cpu().numpy()
        tensor = np.transpose(tensor, [0, 2, 3, 1])
        return tensor


    def inpaint(self, imgs, masks):
        print("inpainting images ...")
        imgs = self.to_numpy(imgs)
        for i in range(imgs.shape[0]):
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)

        imgs = np.transpose(imgs, [0,3,1,2])
        imgs = torch.from_numpy(imgs)

        imgs = imgs.to(self.device)
        masks = masks.to(self.device)
        imgs = imgs.float().div(255)
        masks = masks.float().div(255)

        imgs_miss = imgs * masks



        result, alpha, raw = self.model(imgs_miss, masks)
        result, alpha, raw = result[0], alpha[0], raw[0]
        result = imgs * masks + result * (1 - masks)

        return result


train_path = '/lab/vislab/DATA/CUB/images/justin-train/'
valid_path = '/lab/vislab/DATA/CUB/images/justin-test/'

class Basic(LightningModule):
    def __init__(self, hparams, trial, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.trial = trial

        self.epoch = 0
        self.learning_rate = self.hparams.lr
        self.model = models.resnet50(pretrained=True)
        self.loss = losses.TripletMarginLoss(margin=0.1, triplets_per_anchor="all", normalize_embeddings=True)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model(x)

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        self.trainset = datasets.ImageFolder(train_path, transform)
        self.validset = datasets.ImageFolder(valid_path, transform)

    def train_dataloader(self):
        train_sampler = samplers.MPerClassSampler(self.trainset.targets, 8, len(self.trainset))
        return DataLoader(self.trainset, batch_size=32, sampler=train_sampler, num_workers=4)

    def val_dataloader(self):
        valid_sampler = samplers.MPerClassSampler(self.validset.targets, 8, len(self.validset))
        return DataLoader(self.validset, batch_size=32, sampler=valid_sampler, num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        mask_size = hparams['mask_size_suggestion']
        print(mask_size)

        location = [(a,b) for a in range(255) for b in range(255)]
        small_mask = Image.new('L', (mask_size, mask_size), 0) # self.mask_size_suggestion
        masks = []
        for _ in range(32):
            base = Image.new('L',(256,256),255)
            r = random.choice(location)
            base.paste(small_mask, r)
            location.pop(location.index(r))
            base = np.ascontiguousarray(np.expand_dims(base, 0)).astype(np.uint8)
            masks.append(base)
        masks = np.array(masks)
        masks = torch.from_numpy(masks)

        pretrained_model_path = '/lab/vislab/DATA/just/infilling/model/model_places2.pth'
        inpainter = Inpainter(pretrained_model_path, 256, 32)
        inpainted_img_batch = inpainter.inpaint(inputs, masks)
        inputs = inputs.to('cpu')
        inputs = 0 
        del inputs
        print("inpainted type", type(inpainted_img_batch))
        print("inpainted batch shape", inpainted_img_batch.shape)

        outputs = self(inpainted_img_batch)
        loss = self.loss(outputs, labels)

        masks = 0 
        del masks
        del inpainted_img_batch

        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'train_loss': train_loss},
            'progress_bar': {'train_loss': train_loss}
        }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        # calculate that stepped accuracy
        val_acc = self.computeAccuracy(outputs, labels)
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        # needs to be a tensor

        return {
            'val_loss': loss, 
            'val_acc': val_acc
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        # stack the accuracies and average PER EPOCH
        val_acc = torch.stack([x['val_acc'] for x in validation_step_outputs]).mean()
        # print(validation_step_outputs['val_acc'])
        print("val_acc" , val_acc)
        print("val_loss", val_loss)

        self.trial.report(val_loss, self.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        self.epoch += 1
        return {
            'log': {
                'val_loss': val_loss,
                'val_acc': val_acc
                },
            'progress_bar': {
                'val_loss': val_loss,
                'val_acc': val_acc
            }
        }
    
    def computeAccuracy(self, outputs, labels):
        incorrect = correct = 0
        for idx, emb in enumerate(outputs):
            pairwise = torch.nn.PairwiseDistance(p=2)
            dist = pairwise(emb, outputs)
            closest = torch.topk(dist, 2, largest=False).indices[1]
            if labels[idx] == labels[closest]:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.1)
        return parser