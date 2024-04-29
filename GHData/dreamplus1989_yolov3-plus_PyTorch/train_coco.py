from __future__ import division

from utils.cocoapi_evaluator import COCOAPIEvaluator
from data import *
from utils.augmentations import SSDAugmentation
import tools

import os
import random
import argparse
import time
import math
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

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
    parser.add_argument('-fl', '--use_focal', action='store_true', default=False,
                        help='use focal loss')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-ciou', '--ciou_loss', action='store_true', default=False,
                        help='use ciou to regress bbox.')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--num_classes', default=80, type=int, 
                        help='The number of dataset classes')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/coco/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()
    data_dir = coco_root

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False  
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    
    cfg = coco_ab

    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.mosaic:
        print("use Mosaic Augmentation ...")

    # multi scale
    if args.multi_scale:
        print('Let us use the multi-scale trick.')
        input_size = [640, 640]
    else:
        input_size = [416, 416]

    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the MSCOCO dataset...')
    # dataset
    dataset = COCODataset(
                data_dir=data_dir,
                img_size=input_size[0],
                transform=SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                debug=args.debug,
                mosaic=args.mosaic)

    # data loader
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers)

    # cocoapi evaluator
    evaluator = COCOAPIEvaluator(
                    data_dir=data_dir,
                    img_size=cfg['min_dim'],
                    device=device,
                    transform=BaseTransform(cfg['min_dim'])
                    )
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    # # yolo_v3_plus series: yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small
    if args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-53'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_plus on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_large':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-l'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_plus_large on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_half':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-h'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_plus_half on the COCO dataset ......')
        
    elif args.version == 'yolo_v3_plus_medium':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-m'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_plus_medium on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_small':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-s'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_plus_small on the COCO dataset ......')
    
    # # yolo_v3_slim series: 
    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_slim on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-slim'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_slim_csp on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp2':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo_v3_slim_csp2 on the COCO dataset ......')
        
    # # yolo_v3_spp
    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-53'
        
        yolo_net = YOLOv3SPP(device, input_size=input_size, num_classes=args.num_classes, trainable=True, anchor_size=anchor_size, hr=hr, backbone=backbone, ciou=args.ciou_loss)
        print('Let us train yolo-v3-spp on the COCO dataset ......')

    else:
        print('Unknown version !!!')
        exit()

    model = yolo_net
    model.to(device).train()

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))
 
    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):

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

        for iter_i, (images, targets) in enumerate(dataloader):
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
            conf_loss, cls_loss, bbox_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), bbox_loss.item(), total_loss.item(), input_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')
                        )  
    
        # COCO evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            model.set_grid(cfg['min_dim'])
            # evaluate
            ap50_95, ap50 = evaluator.evaluate(model)
            print('ap50 : ', ap50)
            print('ap50_95 : ', ap50_95)
            # convert to training mode.
            model.trainable = True
            model.set_grid(input_size)
            model.train()
            if args.tfboard:
                writer.add_scalar('val/COCOAP50', ap50, epoch + 1)
                writer.add_scalar('val/COCOAP50_95', ap50_95, epoch + 1)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    h, w = input_size

    batch_size = images.size(0)
    for i in range(batch_size):
        img = images[i].clone().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        img = ((img * std + mean)*255).astype(np.uint8)
        cv2.imwrite('1.jpg', img)
        img_ = cv2.imread('1.jpg')
        target = targets[i].clone()

        for box in target:
            xmin, ymin, xmax, ymax = box[:-1]
            # print(xmin, ymin, xmax, ymax)
            xmin *= w
            ymin *= h
            xmax *= w
            ymax *= h
            cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

        cv2.imshow('img', img_)
        cv2.waitKey(0)


if __name__ == '__main__':
    train()
