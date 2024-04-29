from data import *
from utils.augmentations import SSDAugmentation
import torch.backends.cudnn as cudnn
import os
import time
import math
import random
import tools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo_v3_spp',
                        help='yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_half, yolo_v3_plus_medium, yolo_v3_plus_small, \
                              yolo_v3_slim, yolo_v3_slim_csp, yolo_v3_slim_csp2, \
                              yolo_v3_spp.')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO dataset')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation.')
    parser.add_argument('--dataset_root', default=VOC_ROOT, 
                        help='Location of VOC root directory')
    parser.add_argument('--num_classes', default=20, type=int, 
                        help='The number of dataset classes')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='weights/voc/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--resume', type=str, default=None,
                        help='fine tune the model trained on MSCOCO.')

    return parser.parse_args()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# setup_seed(20)
def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False  
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    
    cfg = voc_ab

    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.mosaic:
        print("use Mosaic Augmentation ...")
        
    # use multi-scale trick
    # multi scale
    if args.multi_scale:
        print('Let us use the multi-scale trick.')
        input_size = [640, 640]
    else:
        input_size = [416, 416]

    # dataset
    dataset = VOCDetection(root=args.dataset_root, img_size=input_size[0],
                            transform=SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                            mosaic=args.mosaic)

    # build model
    # # yolo_v3_plus series: yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small
    if args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'd-53'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_plus on the VOC dataset ......')
    
    elif args.version == 'yolo_v3_plus_large':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'csp-l'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_plus_large on the VOC dataset ......')
    
    elif args.version == 'yolo_v3_plus_half':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'csp-h'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_plus_half on the VOC dataset ......')
        
    elif args.version == 'yolo_v3_plus_medium':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'csp-m'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_plus_medium on the VOC dataset ......')
    
    elif args.version == 'yolo_v3_plus_small':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'csp-s'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_plus_small on the VOC dataset ......')
    
    # # yolo_v3_slim series: 
    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'd-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_slim on the VOC dataset ......')

    elif args.version == 'yolo_v3_slim_csp':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'csp-slim'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_slim_csp on the VOC dataset ......')

    elif args.version == 'yolo_v3_slim_csp2':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'csp-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo_v3_slim_csp2 on the VOC dataset ......')

    # # yolo_v3_spp
    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        anchor_size = MULTI_ANCHOR_SIZE
        backbone = 'd-53'
        
        yolo_net = YOLOv3SPP(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone)
        print('Let us train yolo-v3-spp on the VOC dataset ......')

    else:
        print('Unknown version !!!')
        exit()


    # finetune the model trained on COCO 
    if args.resume is not None:
        print('finetune COCO trained ')
        yolo_net.load_state_dict(torch.load(args.resume, map_location=device), strict=False)


    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/voc/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    print("----------------------------------------Object Detection--------------------------------------------")
    model = yolo_net
    model.to(device)

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    # loss counters
    print("----------------------------------------------------------")
    print("Let's train OD network !")
    print('Training on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    epoch_size = len(dataset) // args.batch_size
    max_epoch = cfg['max_epoch']

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    t0 = time.time()

    # start training
    for epoch in range(max_epoch):
        
        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(data_loader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
                    
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                input_size = [size, size]
                model.set_grid(input_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=False)

            # make train label
            targets = [label.tolist() for label in targets]
            targets = tools.multi_gt_creator(input_size, yolo_net.stride, targets, anchor_size=anchor_size)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)
                     
            # backprop and update
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), input_size[0], t1-t0),
                        flush=True)

                t0 = time.time()


                # change input dim
                # But this operation will report bugs when we use more workers in data loader, so I have to use 0 workers.
                # I don't know how to make it suit more workers, and I'm trying to solve this question.
                # data_loader.dataset.reset_transform(SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')  
                    )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

if __name__ == '__main__':
    train()