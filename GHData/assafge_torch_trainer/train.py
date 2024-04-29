#!/usr/bin/env python
import argparse
try:
    from torch_trainer.trainer import TorchTrainer
except ModuleNotFoundError:
    from trainer import TorchTrainer


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch training module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    # parser.add_argument('-v', '--debug_prints', help='path to output directory')
    new = parser.add_argument_group('new model')
    new.add_argument('-m', '--model_cfg', help='path to model cfg file')
    new.add_argument('-o', '--optimizer_cfg', help='path to optimizer cfg file')
    new.add_argument('-d', '--dataset_cfg', help='path to dataset cfg file')
    new.add_argument('-w', '--out_path', help='path to output directory')
    new.add_argument('-e', '--exp_name', default='')
    pre_train = parser.add_argument_group('pre-trained')
    pre_train.add_argument('-r', '--model_path', help='path to pre-trained model')
    pre_train.add_argument('-s', '--non_strict', help='set to false to use pre-trained parts from another model',
                           action='store_false', default=True)


    args = parser.parse_args()
    return args


def main():

    args = get_args()

    if args.model_path:
        trainer = TorchTrainer.warm_startup(in_path=args.model_path, gpu_index=args.gpu_index)
    else:
        trainer = TorchTrainer.new_train(out_path=args.out_path, model_cfg=args.model_cfg, optimizer_cfg=args.optimizer_cfg,
                                         dataset_cfg=args.dataset_cfg, gpu_index=args.gpu_index, exp_name=args.exp_name)
    trainer.train()


if __name__ == '__main__':
    main()
