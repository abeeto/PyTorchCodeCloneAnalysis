import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn
from torch.utils.data import DataLoader

from torchvision import transforms

from core.resnet38_cls import Classifier

from tools.utils import *
from tools.txt_utils import *
from tools.dataset_utils import *
from tools.augment_utils import *
from tools.torch_utils import *
from tools.ca_utils import *

###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--architecture', default='resnet38', type=str)
parser.add_argument('--pretrained', default=True, type=str2bool)

parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epoches', default=15, type=int)

parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--weight_decay', default=5e-5, type=float)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--print_ratio', default=0.01, type=float)

parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--num_combination', default=2, type=int)

parser.add_argument('--root_dir', default='F:/VOCtrainval_11-May-2012/', type=str)

args = parser.parse_args()
###############################################################################

if __name__ == '__main__':
    experiment_name = 'VOC2012'
    experiment_name += '@arch={}'.format(args.architecture)
    experiment_name += '@pretrained={}'.format(args.pretrained)
    experiment_name += '@alpha={}'.format(args.alpha)
    experiment_name += '@num_combination={}'.format(args.num_combination)
    experiment_name += '@lr={}'.format(args.learning_rate)
    experiment_name += '@wd={}'.format(args.weight_decay)
    experiment_name += '@bs={}'.format(args.batch_size)
    experiment_name += '@epoch={}'.format(args.max_epoches)

    model_dir = create_directory('./models/')
    model_path = model_dir + f'{experiment_name}.pth'

    cudnn.enabled = True

    train_transform = transforms.Compose([
        RandomResize(256, 512),
        RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

        np.asarray,
        Normalize(),

        RandomCrop(args.image_size),

        Transpose2D(),
        torch.from_numpy
    ])
    test_transform = transforms.Compose([
        np.asarray,
        Normalize(),

        CenterCrop(args.image_size),

        Transpose2D(),
        torch.from_numpy
    ])

    class_names = read_txt('./data/class_names.txt')
    class_dic = {class_name:class_index for class_index, class_name in enumerate(class_names)}

    model = Classifier(len(class_names))

    class_loss_fn = F.multilabel_soft_margin_loss
    feature_loss_fn = nn.L1Loss().cuda()

    train_dataset = VOC_Dataset(args.root_dir, './data/train_aug.txt', class_dic, train_transform)
    valid_dataset = VOC_Dataset(args.root_dir, './data/val.txt', class_dic, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = val_iteration * args.max_epoches

    print('[i] log_iteration : {:,}'.format(log_iteration))
    print('[i] val_iteration : {:,}'.format(val_iteration))
    print('[i] max_iteration : {:,}'.format(max_iteration))

    param_groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': param_groups[1], 'lr': 2*args.learning_rate, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': param_groups[3], 'lr': 20*args.learning_rate, 'weight_decay': 0}
    ], lr=args.learning_rate, weight_decay=args.weight_decay, max_step=max_iteration)

    if args.pretrained:
        if args.architecture == 'resnet38':
            weights_dict = torch.load('./pretrained_models/resnet-34.pth')
            model.load_state_dict(weights_dict, strict=False)
            print('[i] load model weights')

    model = model.cuda()
    model.train()

    train_timer = Timer()
    valid_timer = Timer()

    train_meter = AverageMeter(['loss', 'class_loss', 'single_class_loss', 'feature_loss'])
    train_iterator = Iterator(train_loader)

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        the_number_of_gpu = len(use_gpu.split(','))
        if the_number_of_gpu > 1:
            print('[i] the number of gpu : {}'.format(the_number_of_gpu))
            model = nn.DataParallel(model)
    except KeyError:
        pass

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

    def validate(model, loader):
        valid_meter = AverageMeter(['loss'])

        model.eval()

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                logits = model(images)
                loss = class_loss_fn(logits, labels)

                valid_meter.add({'loss': loss.item()})

        model.train()
        return valid_meter.get(clear=True)

    best_valid_loss = -1

    for iteration in range(1, max_iteration + 1):
        images, labels = train_iterator.get()
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        # for image, label in zip(images, labels):
        #     print(image.size(), label.size())
        # input()

        logits, features = model(images, with_cam=True)

        # calculate single class loss
        single_class_loss = class_loss_fn(logits, labels)

        ###############################################################################
        # Combined Augmentation
        ###############################################################################
        the_size_of_reduction = args.num_combination**2

        _, _, h, w = images.size()
        splitted_images = [
            F.interpolate(x, size=(h // args.num_combination, w // args.num_combination), mode='bilinear', align_corners=False)
                for x in torch.split(images, args.batch_size // the_size_of_reduction, dim=0)]
        splitted_labels = torch.split(labels, labels.size()[0] // the_size_of_reduction, dim=0)
        
        combined_images = combine_2D_for_images(splitted_images, args.num_combination, use_cuda=True)
        _, combined_features = model(combined_images, with_cam=True)
        decomposed_features = decompose_2D_for_features(combined_features, args.num_combination)

        # calculate combined class loss
        combined_class_loss = class_loss_fn(global_average_pooling_2d(decomposed_features[0], with_flatten=True), splitted_labels[0])
        for i in range(1, the_size_of_reduction):
            combined_class_loss += class_loss_fn(global_average_pooling_2d(decomposed_features[i], with_flatten=True), splitted_labels[i])
        
        combined_class_loss += class_loss_fn(global_average_pooling_2d(combined_features, with_flatten=True), combine_for_labels(splitted_labels))
        combined_class_loss /= (the_size_of_reduction + 1)

        combined_features = torch.cat(decomposed_features, dim=0)
        feature_loss = feature_loss_fn(F.interpolate(combined_features, size=tuple(features.size()[2:]), mode='bilinear', align_corners=False), features)
        
        # merge class loss
        class_loss = single_class_loss + combined_class_loss

        # print(class_loss, single_class_loss, combined_class_loss, feature_loss)

        loss = class_loss + args.alpha * feature_loss
        ###############################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'class_loss' : class_loss.item(), 
            'single_class_loss' : single_class_loss.item(),
            'feature_loss' : feature_loss.item(), 
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if iteration % log_iteration == 0:
            loss, class_loss, single_class_loss, feature_loss = train_meter.get(clear=True)
            learning_rate = get_learning_rate_from_optimizer(optimizer)
            
            data = {
                'iteration' : iteration,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'class_loss' : class_loss,
                'single_class_loss' : single_class_loss,
                'feature_loss' : feature_loss,
                'time' : train_timer.tok(clear=True),
            }

            print('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                single_class_loss={single_class_loss:.4f}, \
                feature_loss={feature_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )
            
        #################################################################################################
        # Validation
        #################################################################################################
        if iteration % val_iteration == 0:
            valid_timer.tik()
            valid_loss = validate(model, valid_loader)
            
            if best_valid_loss == -1 or best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                save_model_fn()
                print('[i] save model')

            data = {
                'iteration' : iteration,
                'valid_loss' : valid_loss,
                'best_valid_loss' : best_valid_loss,
                'time' : valid_timer.tok(clear=True),
            }

            print('[i] \
                iteration={iteration:,}, \
                valid_loss={valid_loss:.4f}, \
                best_valid_loss={best_valid_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )
            