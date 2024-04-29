import math
import argparse

import numpy as np
from tqdm import tqdm

import eval_core
import flops_counter
import models
from data import WIDERFace


def parseargs():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser('FLOPs Evaluation for Models on WIDER Face.')

    parser.add_argument('--widerface_root',
                        default='data/widerface',
                        help='Root of WIDER Face dataset.')
    parser.add_argument('--split',
                        default='val',
                        choices=['val-easy', 'val-medium', 'val-hard', 'val', 'test'],
                        help='Run FLOPs evaluation on WIDER Face validation set or test set.')
    parser.add_argument('--model',
                        default='CSP',
                        help='Evaluate the specified model.')
    parser.add_argument('--ms',
                        default=True,
                        type=str2bool,
                        help='Set to true to evaluate in multi scales or false to evaluate in single scale 1.')
    parser.add_argument('--max_downsample',
                        default=1,
                        type=int,
                        help='Make the input size dividable by given value due to model design.')

    args = parser.parse_args()
    return args

def to_scientific(flops):
    p = math.floor(math.log(flops, 2))
    rank = p // 10
    left = p % 10
    dim = ''
    if rank == 1:
        dim = 'K'
    elif rank == 2:
        dim = 'M'
    elif rank == 3:
        dim = 'G'
    elif rank == 4:
        dim = 'P'
    v = flops
    for i in range(rank * 10):
        v = v / 2
    return '{:.4f} {}'.format(v, dim)

def main(args):
    # build dataset
    dataset = WIDERFace(widerface_root=args.widerface_root, split=args.split)

    model_name = ''
    version = None
    _model = args.model.lower().split('-')
    model_name = _model[0]
    if len(_model) == 2:
        version = _model[-1]

    available_models = [
        'CSP', # NOTE: max_downsample must be 16
        'DSFD',
        'EXTD', # EXTD-32/48/64
        'FaceBoxes',
        'HR',
        'LFFD', # LFFD-v1/v2
        'light_DSFD',
        'PyramidBox',
        'RetinaFace',
        'S3FD',
        'SFA',
        'SHF',
        'SRN',
        'SSH',
        'ULFG',
        'YuFaceDetectNet'
    ]
    for m in available_models:
        if m.lower() == model_name:
            model_name = m
            break

    # build model
    kwargs = [version] if version else []
    model = eval('models.'+model_name)(*kwargs)

    # calculate average flops
    try:
        eval_handle = eval('eval_core.'+model_name)(dataset, model, args.ms, args.max_downsample)
    except:
        eval_handle = eval_core.Eval(dataset, model, max_downsample=args.max_downsample)
    avg_flops = eval_handle.test()
    print(avg_flops)
    print('Average FLOPs: {}FLOPs'.format(
        to_scientific(avg_flops)
    ))


if __name__ == '__main__':
    args = parseargs()
    main(args)