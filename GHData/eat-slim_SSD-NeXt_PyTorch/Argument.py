import argparse

parser_SSD_NeXt = argparse.ArgumentParser(description='Train SSDv3')

parser_SSD_NeXt.add_argument('--batch_size', default=23, type=int,
                             help='Batch size for training')
parser_SSD_NeXt.add_argument('--sample_per_step', default=640, type=int,
                             help='Batch number contained in a step')
parser_SSD_NeXt.add_argument('--num_epoch', default=80, type=int,
                             help='Number of epoch for training')
parser_SSD_NeXt.add_argument('--num_workers', default=14, type=int,
                             help='Number of workers used in dataloader')
parser_SSD_NeXt.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                             help='initial learning rate for optimizer')
parser_SSD_NeXt.add_argument('--weight_decay', default=5e-4, type=float,
                             help='Weight decay for optimizer')
parser_SSD_NeXt.add_argument('--momentum', default=0.9, type=float,
                             help='Momentum value for optimizer')
parser_SSD_NeXt.add_argument('--warm_up', default=5, type=int,
                             help='Number of warm up epochs')
parser_SSD_NeXt.add_argument('--unfreeze', default=0, type=int,
                             help='Number of freeze epochs')
parser_SSD_NeXt.add_argument('--save_folder', default='weights', type=str,
                             help='Directory for saving checkpoint models')
parser_SSD_NeXt.add_argument('--name', default='SSD_NeXt', type=str,
                             help='Name of the model')
parser_SSD_NeXt.add_argument('--version', default='AdamW', type=str,
                             help='Version of the model')
parser_SSD_NeXt.add_argument('--eval_frequency', default=1, type=int,
                             help='Eval every few epochs')
parser_SSD_NeXt.add_argument('--checkpoint', default='', type=str,
                             help='Path to checkpoint file')
parser_SSD_NeXt.add_argument('--dataset', default='VOC', type=str,
                             help='Dataset for training')
parser_SSD_NeXt.add_argument('--set_lr', default=0, type=float,
                             help='Set learning rate as you want, not initial lr')
parser_SSD_NeXt.add_argument('--test_bsz', default=8, type=float,
                             help='Batch size for test')
parser_SSD_NeXt.add_argument('--mode', default='train', type=str,
                             help='train or test')

parser_SSD300 = argparse.ArgumentParser(description='Train SSD300')

parser_SSD300.add_argument('--batch_size', default=32, type=int,
                           help='Batch size for training')
parser_SSD300.add_argument('--sample_per_step', default=640, type=int,
                           help='Batch number contained in a step')
parser_SSD300.add_argument('--num_epoch', default=120, type=int,
                           help='Number of epoch for training')
parser_SSD300.add_argument('--num_workers', default=14, type=int,
                           help='Number of workers used in dataloader')
parser_SSD300.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                           help='initial learning rate for optimizer')
parser_SSD300.add_argument('--weight_decay', default=5e-4, type=float,
                           help='Weight decay for optimizer')
parser_SSD300.add_argument('--momentum', default=0.9, type=float,
                           help='Momentum value for optimizer')
parser_SSD300.add_argument('--warm_up', default=5, type=int,
                           help='Number of warm up epochs')
parser_SSD300.add_argument('--unfreeze', default=0, type=int,
                           help='Number of freeze epochs')
parser_SSD300.add_argument('--save_folder', default='weights', type=str,
                           help='Directory for saving checkpoint models')
parser_SSD300.add_argument('--name', default='SSD300', type=str,
                           help='Name of the model')
parser_SSD300.add_argument('--version', default='AdamW', type=str,
                           help='Version of the model')
parser_SSD300.add_argument('--eval_frequency', default=1, type=int,
                           help='Eval every few epochs')
parser_SSD300.add_argument('--checkpoint', default='', type=str,
                           help='Path to checkpoint file')
parser_SSD300.add_argument('--dataset', default='VOC', type=str,
                           help='Dataset for training')
parser_SSD300.add_argument('--set_lr', default=0, type=float,
                           help='Set learning rate as you want, not initial lr')
parser_SSD300.add_argument('--test_bsz', default=8, type=float,
                           help='Batch size for test')
parser_SSD300.add_argument('--mode', default='train', type=str,
                           help='train or test')

parameters_group = {
    'p2': ({
               'u_xy': 0,
               'u_wh': 0,
               'sigma_xy': 0.1,
               'sigma_wh': 0.2,
               'eps': 1e-6,
               'cardinal_eps': 1e-3,
               'modified': False
           },
           {
               'small_boxes_iou_ratio': 0.5,
               'weighting': 3
           }),
    'p1': ({
               'u_xy': 0,
               'u_wh': 0,
               'sigma_xy': 0.1,
               'sigma_wh': 0.2,
               'eps': 1e-6,
               'cardinal_eps': 1e-3,
               'modified': True
           },
           {
               'small_boxes_iou_ratio': 0.6,
               'weighting': 2
           })
}
parameters = parameters_group['p1']
encoding_parameters = parameters[0]
match_parameters = parameters[1]


