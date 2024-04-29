from __future__ import print_function

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from lib.nn import async_copy_to

category_merge_list = [
    [22, 27, 61, 129],                              # Water class: ’water’, ’sea’, ’river’, ’lake’
    [18, 10, 73, 5, 33],                            # Tree class: ’plant’, ’grass’, ’tree’, ’fence’
    [26, 2, 48, 1, 80, 85],                         # Building class: ’house’, ’building’, ’skyscraper’, ’wall’, ’hovel’, ’tower’
    [7, 53, 12, 47, 69, 14, 30, 95, 52, 102, 92],   # Road class: ’road’, ’path’, ’sidewalk’, ’sand’, ’hill’, ’earth’, ’field’, ’land’, ’grandstand’, ’stage’, ’dirt track’
    [87, 6],                                        # Roof class: ’awning’, ’ceiling’
    [17, 35],                                       # Mountain class: ’mountain’, ’rock’
    [60, 54, 122],                                  # Stair class: ’stairway’, ’stairs’, ’step’
    [76, 20, 31, 32, 24, 70],                       # Chair class: "chair’, ’seat’, ’armchair’, ’sofa’, ’bench’, ’swivel chair’
    [21, 81, 84, 103]                               # Vehicle class: ’car’, ’bus’, ’truck’, ’van’
]

def segment_img(net, data, seg_size, args, valid_masks=None, cutoff=0.2):
    """
    return Tensor (Categories, H, W)
    """
    img_resized_list = data['img_data']
    pred = torch.zeros(1, args.num_class, seg_size[0], seg_size[1])
    for img in img_resized_list:
        feed_dict = data.copy()
        feed_dict['img_data'] = img
        del feed_dict['img_ori']
        del feed_dict['info']
        feed_dict = async_copy_to(feed_dict, 0)

        # forward pass
        pred_tmp = net(feed_dict, segSize=seg_size)
        pred = pred + pred_tmp.cpu() / len(args.imgSize)

    if valid_masks is not None:
        mask = torch.zeros(1, args.num_class, seg_size[0], seg_size[1])
        mask[:, valid_masks, :, :] = 1
        pred *= mask
        pred = pred / (pred.sum(dim=1) + 1e-6)

    # cut off
    pred[pred < cutoff] = 0
    return pred.detach().squeeze()

def test(segmentation_module, data, seg_size, args):
    tar_seg = segment_img(segmentation_module, data["tar"], seg_size, args)
    valid_categories = np.unique(tar_seg.numpy().nonzero()[0])
    # input image can only be segmented with categories in target image
    in_seg = segment_img(segmentation_module, data["in"], seg_size, args, valid_categories)

    # merge categories
    for cat in category_merge_list:
        cat = np.array(cat) - 1     # convert to 0-index
        tar_seg[cat[0]] = tar_seg[cat].sum(dim=0)
        tar_seg[cat[1:]] = 0
        in_seg[cat[0]] = in_seg[cat].sum(dim=0)
        in_seg[cat[1:]] = 0
    
    # only keep valid category layers
    valid_categories = np.unique(tar_seg.numpy().nonzero()[0])
    in_seg = in_seg[valid_categories]
    tar_seg = tar_seg[valid_categories]
    print("Categories: ", valid_categories + 1) # convert to 1-index for showing

    return {"in": in_seg, "tar": tar_seg, "categories": valid_categories}

def load_data(data_dict, args):
    data_list = [{'fpath_img': data_dict["in"]}, {'fpath_img': data_dict["tar"]}]
    dset = TestDataset(data_list, args)
    return {"in": dset[0], "tar": dset[1]}

def segment(args, h, w):
    # absolute paths of model weights
    weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(weights_encoder) and \
        os.path.exists(weights_encoder), 'checkpoint does not exitst!'

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()

    # Dataset and Loader
    data_dict = {"in": args.content, "tar": args.style}
    data = load_data(data_dict, args)

    # Main loop
    with torch.no_grad():
        res = test(segmentation_module, data, (h, w), args)
    print('Inference done!')
    return res
    

def main(args):
    res = segment(args, args.height, args.width)
    torch.save(res, args.save_path)


def add_arguments(parser):
    parser.add_argument('--model_path', default='models',
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--content', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument("--height", type=int, default=480, help="Size for scaling image.")
    parser.add_argument("--width", type=int, default=720, help="Size for scaling image.")
    add_arguments(parser)
    args = parser.parse_args()
    print(args)

    main(args)
