#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import time
import models as models
import models_cpu as models_cpu
import copy
import random
import argparse
import numpy as np
import tensorly as tl

tl.set_backend('pytorch')
import torch.backends.cudnn as cudnn
from flopco import FlopCo
import gc
from layers.cpd import CPD3_layer
from layers.svd import SVD_conv_layer
from utils.replacement_utils import (get_layer_by_name, replace_conv_layer_by_name,
                                     batchnorm_callibration)

from decompose.cp_decomposition import cp_decompose_layer
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Layer Decomposer")
parser.add_argument('--layer', default='', type=str, required=True, help='(default=%(default)s)')
parser.add_argument('--rank', default='', type=int, required=True, help='(default=%(default)s)')
parser.add_argument('--eps', default='0.002', type=float, required=True, help='(defauls=%(default)s)')
parser.add_argument('--dpath', default='', type=str, required=True, help='Dataset Directory')
parser.add_argument('--mpath', default='', type=str, required=True, help='Trained Model file(.pth)')
parser.add_argument('--tlabels', default='', type=str, required=True, help='Labels for Train Set')
parser.add_argument('--vlabels', default='', type=str, required=True, help='Labels for Test/validation Set')
parser.add_argument('--ranks_dir', default='../ranks_PA-100K/', type=str, required=False,
                    help='Dir to save ranks and CPD-Tensors')
parser.add_argument('--device', default='cuda', type=str, required=False, help='device to use for decompositions')
parser.add_argument('--attr_num', default='26', type=int, required=True, help='(35)PETA or (26)PA-100K')
parser.add_argument('--experiment', default='PA-100K', type=str, required=True,
                    help='Type of experiment PETA or PA-100K')
args = parser.parse_args()

data_path = args.dpath
train_list_path = args.tlabels
val_list_path = args.vlabels
# experiment = args.experiment
lnames_to_compress = args.layer
max_ranks = args.rank
eps_c = args.eps
attr_num = args.attr_num
EPS = 1e-12


####DATA LOADER CLASS###

def default_loader(path):
    return Image.open(path).convert('RGB')


class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
                images
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)
    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()
    res = []
    for k in range(attr_num):
        res.append(1.0 * sum(correct[:, k]) / batch_size)
    return sum(res) / attr_num


##VALIDATE FUNCTION
def validate(val_loader, model, criterion):
    """Perform validation on the validation set"""
    print_freq = 100
    total_len = len(val_loader)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    for i, _ in enumerate(val_loader, start=1):
        input, target = _
        target = target.to(device)
        input = input.to(device)
        output = model(input)

        bs = target.size(0)

        if type(output) == type(()) or type(output) == type([]):
            loss_list = []
            # deep supervision
            for k in range(len(output)):
                out = output[k]
                loss_list.append(criterion.forward(torch.sigmoid(out), target))
            loss = sum(loss_list)
            # maximum voting
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
        else:
            loss = criterion.forward(torch.sigmoid(output), target)

        # measure accuracy and record loss
        accu = accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, total_len, batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Accu {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


###Loss Function Class
class Weighted_BCELoss(object):
    """
        Weighted_BCELoss was proposed in "Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios"[13].
    """

    def __init__(self, experiment):
        super(Weighted_BCELoss, self).__init__()
        self.weights = None
        if experiment == 'peta' or experiment == 'PETA':
            self.weights = torch.Tensor([0.5016,
                                         0.3275,
                                         0.1023,
                                         0.0597,
                                         0.1986,
                                         0.2011,
                                         0.8643,
                                         0.8559,
                                         0.1342,
                                         0.1297,
                                         0.1014,
                                         0.0685,
                                         0.314,
                                         0.2932,
                                         0.04,
                                         0.2346,
                                         0.5473,
                                         0.2974,
                                         0.0849,
                                         0.7523,
                                         0.2717,
                                         0.0282,
                                         0.0749,
                                         0.0191,
                                         0.3633,
                                         0.0359,
                                         0.1425,
                                         0.0454,
                                         0.2201,
                                         0.0178,
                                         0.0285,
                                         0.5125,
                                         0.0838,
                                         0.4605,
                                         0.0124]).to(device)
        elif experiment == "PA-100K" or experiment == "pa-100k":
            self.weights = torch.Tensor([0.460444444444,
                                         0.0134555555556,
                                         0.924377777778,
                                         0.0621666666667,
                                         0.352666666667,
                                         0.294622222222,
                                         0.352711111111,
                                         0.0435444444444,
                                         0.179977777778,
                                         0.185,
                                         0.192733333333,
                                         0.1601,
                                         0.00952222222222,
                                         0.5834,
                                         0.4166,
                                         0.0494777777778,
                                         0.151044444444,
                                         0.107755555556,
                                         0.0419111111111,
                                         0.00472222222222,
                                         0.0168888888889,
                                         0.0324111111111,
                                         0.711711111111,
                                         0.173444444444,
                                         0.114844444444,
                                         0.006]).to(device)

    def forward(self, output, target):
        if self.weights is not None:
            cur_weights = torch.exp(target + (1 - target * 2) * self.weights)
            loss = cur_weights * (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
        else:
            loss = target * torch.log(output + EPS) + (1 - target) * torch.log(1 - output + EPS)
        return torch.neg(torch.mean(loss))


train_dataset = MultiLabelDataset(root=data_path, label=train_list_path, transform=transform_train)
val_dataset = MultiLabelDataset(root=data_path, label=val_list_path, transform=transform_test)

BN_CAL_ITERS = 300  # 512000 // BATCH_SIZE

GRID_STEP = 1
nx = 1  # minimal compression ratio

BATCH_SIZE = 32
device = args.device
# device = 'cpu'
NUM_THREADS = 16
MAX_ITERS = 500
TOL = 1e-6

# fix random seed

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
resume = args.mpath

ranks_dir = args.ranks_dir
if not os.path.exists(ranks_dir):
    os.mkdir(ranks_dir)

lr = 0.0001
weight_decay = 0.0005

# Data loading code
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# create model
if device == "cuda":
    model = models.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
else:
    model = models_cpu.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
    new_state_dict = OrderedDict()
    checkpoint = torch.load(resume)
    for i, j in checkpoint['state_dict'].items():
        name = i[7:]
        new_state_dict[name] = j
    model.load_state_dict(new_state_dict)
    model.eval()

if args.experiment == 'peta' or args.experiment == 'PETA':
    criterion = Weighted_BCELoss('peta')
else:
    criterion = Weighted_BCELoss('PA-100K')

optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             betas=(0.9, 0.999),
                             weight_decay=weight_decay)
cudnn.benchmark = False
cudnn.deterministic = True
model_stats = FlopCo(model, img_size=(1, 3, 256, 128), device=device)
all_lnames = list(model_stats.flops.keys())

time_val = time.time()
top1_init = validate(val_loader, model, criterion)
print("total validation time: {}".format(time.time() - time_val))
print('Initial top1 accuracy: {}'.format(top1_init))

saved_ranks = None
min_ranks = 2
curr_rank = max_ranks
curr_max = max_ranks
curr_min = min_ranks

n = int(np.log2(curr_max)) + 1

for i in range(n):
    print("Search iter {}/{}: ranks (min, curr, max): ({}, {}, {})".format(
        i + 1, n, curr_min, curr_rank, curr_max))

    temp_model = copy.deepcopy(model).cpu()
    layer_to_decompose = copy.deepcopy(get_layer_by_name(temp_model, lnames_to_compress))

    print("-------------------------\n Compression step")
    # decompose and replace layer

    if layer_to_decompose.kernel_size[0] == 1:

        decomposed_svd = SVD_conv_layer(layer_to_decompose,
                                        rank_selection='manual',
                                        rank=curr_rank).to(device)
        replace_conv_layer_by_name(temp_model, lnames_to_compress, decomposed_svd)
        del decomposed_svd
        to_save = curr_rank

    else:

        Us_cp = cp_decompose_layer(layer_to_decompose.weight.data, curr_rank,
                                   als_maxiter=MAX_ITERS, als_tol=TOL, epc_tol=TOL,
                                   num_threads=NUM_THREADS, lib='our', use_epc=True)

        to_save = [np.array(U_i) for U_i in Us_cp]
        decomposed_cp = CPD3_layer(layer_to_decompose.to(device), factors=Us_cp,
                                   cpd_type='cp').to(device)
        replace_conv_layer_by_name(temp_model, lnames_to_compress, decomposed_cp)
        del decomposed_cp

    print("-------------------------\n Calibration step")
    # callibrate batch norm statistics
    temp_model.to(device)

    batchnorm_callibration(temp_model, train_loader, layer_name=lnames_to_compress,
                           n_callibration_batches=BN_CAL_ITERS, device=device)

    print("-------------------------\n Test step")
    top1 = validate(val_loader, temp_model, criterion)
    print('Current top1 accuracy: {}'.format(top1))
    del temp_model, layer_to_decompose
    gc.collect()

    if top1 + eps_c < top1_init:

        if i == 0:
            print("Bad layer to compress")
            saved_ranks = curr_rank
            break
        else:
            curr_min = curr_rank
            curr_rank = (curr_max + curr_min) // 2
    else:
        saved_ranks = curr_rank

        curr_max = curr_rank
        curr_rank = (curr_max + curr_min) // 2
        to_save_final = to_save

file_name = 'Inception_ranks_{}_{}_grid_step_{}.npy'.format(lnames_to_compress, eps_c, GRID_STEP)  # svd/cp3?
np.save(ranks_dir + file_name, to_save_final)
