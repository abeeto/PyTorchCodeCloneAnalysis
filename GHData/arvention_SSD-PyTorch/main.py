import os
import argparse
from solver import Solver
from data.data_loader import get_loader
from torch.backends import cudnn
from utils.genutils import mkdir
from datetime import datetime
import zipfile
import torch
import numpy as np


def zipdir(path, ziph):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".py") or file.endswith("cfg"):
            ziph.write(os.path.join(path, file))
            if file.endswith("cfg"):
                os.remove(file)


def save_config(config):
    current_time = str(datetime.now()).replace(":", "_")
    save_name = "ssd_files_{}.{}"
    with open(save_name.format(current_time, "cfg"), "w") as f:
        for k, v in sorted(args.items()):
            f.write('%s: %s\n' % (str(k), str(v)))

    zipf = zipfile.ZipFile(save_name.format(current_time, "zip"),
                           'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()

    return current_time


def str2bool(v):
    return v.lower() in ('true')


def main(version, config):
    # for fast training
    cudnn.benchmark = True

    train_data_loader, test_data_loader = get_loader(config)
    solver = Solver(version, train_data_loader, test_data_loader, vars(config))

    if config.mode == 'train':
        temp_save_path = os.path.join(config.model_save_path, version)
        mkdir(temp_save_path)
        solver.train()
    elif config.mode == 'test':
        if config.dataset == 'voc':
            temp_save_path = os.path.join(config.result_save_path,
                                          config.pretrained_model)
            mkdir(temp_save_path)
        elif config.dataset == 'coco':
            temp_save_path = os.path.join(config.result_save_path, version)
            mkdir(temp_save_path)

        solver.test()


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--class_count', type=int, default=21)
    parser.add_argument('--dataset', type=str, default='voc',
                        choices=['voc', 'coco'])
    parser.add_argument('--new_size', type=int, default=300)
    parser.add_argument('--means', type=tuple, default=(104, 117, 123))
    parser.add_argument('--anchor_config', type=str, default='SSD',
                        choices=['SSD', 'SSD-512',
                                 'ShuffleSSD', 'ShuffleSSD-512'])

    # training settings
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_multiplier', type=int, default=1)
    parser.add_argument('--basenet', type=str,
                        default='vgg16_reducedfc.pth')
    parser.add_argument('--pretrained_model', type=str,
                        default=None)

    # architecture settings
    parser.add_argument('--model', type=str, default='SSD',
                        choices=['SSD', 'FSSD', 'RFBNet',
                                 'ShuffleSSD', 'RShuffleSSD'])
    parser.add_argument('--resnet_model', type=str, default='50',
                        choices=['18', '34', '50', '101'])

    # step size
    parser.add_argument('--counter', type=str, default='iter',
                        choices=['iter', 'epoch'])
    parser.add_argument('--num_iterations', type=int, default=120000)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--loss_log_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=4000)

    # scheduler settings
    parser.add_argument('--warmup', type=str2bool, default=False)
    parser.add_argument('--warmup_step', type=int, default=6)
    parser.add_argument('--sched_milestones', type=list,
                        default=[80000, 100000, 120000])
    parser.add_argument('--sched_gamma', type=float, default=0.1)

    # loss settings
    parser.add_argument('--loss_config', type=str, default='multibox',
                        choices=['multibox', 'focal'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pos_neg_ratio', type=int, default=3)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--use_gpu', type=str2bool, default=True)

    # pascal voc dataset
    parser.add_argument('--voc_config', type=str, default='0712',
                        choices=['0712', '0712+'])
    parser.add_argument('--voc_data_path', type=str,
                        default='../../data/PascalVOC/')

    # coco dataset
    parser.add_argument('--coco_config', type=str, default='2014',
                        choices=['2014', '2017'])
    parser.add_argument('--coco_data_path', type=str,
                        default='../../data/Coco/')

    # path
    parser.add_argument('--model_save_path', type=str, default='./weights')
    parser.add_argument('--result_save_path', type=str, default='./results')

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    version = save_config(config)
    main(version, config)
