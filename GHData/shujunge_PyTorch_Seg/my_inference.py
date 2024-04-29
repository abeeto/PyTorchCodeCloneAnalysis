import warnings
warnings.filterwarnings('ignore')
import os

import torch
import torch.backends.cudnn as cudnn

from models.deeplabv3 import DeepLabV3
from models.unet import UNet
from models.bisenet import BiSeNet
from models.OCNet import OCNet
from models.ICNet import ICNet
from models.PSPNet import PSPNet
from models.danet import DANet
from models.dunet import DUNet
from models.encnet import EncNet
from models.DenseASPP import DenseASPP

import torch.nn as nn
from utils.my_trainer import evalute
from configs.my_argparse import my_argparse
from datasets.VocDataset import VOCSegmentation, make_batch_data_sampler, make_data_sampler
from torchvision import transforms
from torch.utils import data
from utils.loss import MixSoftmaxCrossEntropyLoss, EncNetLoss


if __name__ == "__main__":

    weight_path = "weights"
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    result_path = "results"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # hyper-parameter
    args = my_argparse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUs

    args.model_name = '%dx%d_%s_%s' % (args.image_size, args.image_size, args.backbone, args.head)
    args.nclasses = 21
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ##判断是否有gpu
    if torch.cuda.is_available():
        cudnn.benchmark = True

    print(args)

    Model_Params = {
        'DeepLabV3': {'nclass': args.nclasses, 'stage': args.stage, 'backbone': args.backbone, 'pretrained_base': True},
        'BiSeNet': {'nclass': args.nclasses, 'backbone': args.backbone, 'pretrained_base': True},
        'OCNet': {'nclass': args.nclasses, 'oc_arch': 'pyramid', 'stage': args.stage, 'backbone': args.backbone,
                  'pretrained_base': True},
        'ICNet': {'nclass': args.nclasses, 'backbone': args.backbone, 'pretrained_base': True},
        'DenseASPP': {'nclass': args.nclasses, 'backbone': args.backbone, 'pretrained_base': True},
        'PSPNet': {'nclass': args.nclasses, 'stage': args.stage, 'backbone': args.backbone, 'pretrained_base': True},
        'DANet': {'nclass': args.nclasses, 'stage': args.stage, 'backbone': args.backbone, 'pretrained_base': True},
        'DUNet': {'nclass': args.nclasses, 'backbone': args.backbone, 'pretrained_base': True},
        'EncNet': {'nclass': args.nclasses, 'backbone': args.backbone, 'pretrained_base': True},
        'UNet': {'in_channels': 3, 'n_classes': args.nclasses, 'bilinear': True, 'backbone': args.backbone,
                 'pretrained_base': True, 'usehypercolumns': False},
        }


    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': args.image_size,
                   'root': "/home/zfw/VOCdevkit"}  #
    train_data = VOCSegmentation(split='train', mode='train', **data_kwargs)
    val_data = VOCSegmentation(split='val', mode='val', **data_kwargs)
    iters_per_epoch = len(train_data) // (args.batch_size)
    max_iters = iters_per_epoch

    val_sampler = make_data_sampler(val_data, shuffle=False, distributed=False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

    val_Dataloader = data.DataLoader(dataset=val_data,
                                     batch_sampler=val_batch_sampler,
                                     num_workers=4,
                                     pin_memory=True)

    print("val_dataset:", len(val_data))

    model = eval(args.head)(**Model_Params[args.head]).to(args.device)
    model.load_state_dict(torch.load(args.weight_path))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # model.load_state_dict(torch.load(args.weight_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if args.head == 'ENcNet':
        loss_fn = EncNetLoss(nclass=args.nclasses, ignore_index=-1)
        # elif args.head =='ENcNet':
        # loss_fn = ICNetLoss(nclass=args.nclasses, ignore_index=-1)
    else:
        loss_fn = MixSoftmaxCrossEntropyLoss(aux=False, aux_weight=False,ignore_index=-1)  # [nn.BCEWithLogitsLoss(), DiceLoss()]

    evalute(args, model, loss_fn, val_Dataloader)





