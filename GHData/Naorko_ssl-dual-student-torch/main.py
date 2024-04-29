import logging
import os
import random
import sys
from itertools import product

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from model_executor import execute_model
from src import datasets, data
from src.cli import parse_dict_args, LOG
from src.run_context import RunContext

HP_COMBINATIONS = 49

args = None

algo_short_to_name = {'mt': 'Mean Teacher',
                      'ms': 'Multiple Students',
                      'msi': 'Multiple Students Improved'}


def hp_product():
    bs_hp = [32, 64, 128]
    n_labels_ratio_hp = [0.5, 0.4, 0.25, 0.1]
    wd_hp = [0, 1e-2, 1e-3, 1e-4]
    momentum_hp = [0.3, 0.5, 0.8, 0.9]
    hp_product_lst = list(product(bs_hp, n_labels_ratio_hp, wd_hp, momentum_hp))

    return hp_product_lst


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def partition_dict(dict_of_list, n):
    return {k: partition(lst, n) for k, lst in dict_of_list.items()}


def extract_fold(labeled_dict, unlabeled_dict, fold_idxs):
    def extract_dict(d):
        idxs = []

        for fold_idx in fold_idxs:
            for k in d.keys():
                idxs.extend(d[k][fold_idx])

        return idxs

    return extract_dict(labeled_dict), extract_dict(unlabeled_dict)


def create_loader(dataset, labeled_idxs, unlabeled_idxs, idxs_in_dict, eval=False):
    labeled, unlabeled = extract_fold(labeled_idxs, unlabeled_idxs, idxs_in_dict)

    if not eval:
        sampler = data.TwoStreamBatchSampler(
            unlabeled, labeled, args.batch_size, args.labeled_batch_size)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=True)

    else:
        samples_idxs = labeled + unlabeled
        dataset = Subset(dataset, samples_idxs)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

    return loader


def nested_cross_validation(context, outer_k=10, inner_k=3):
    ncv_log = context.create_train_log('Final_Results')

    # create dataloaders
    dataset_config = datasets.__dict__[args.dataset](tnum=args.model_num)
    num_classes = dataset_config.pop('num_classes')

    train_dataset = torchvision.datasets.ImageFolder(dataset_config['datadir'], dataset_config['train_transformation'])
    eval_dataset = torchvision.datasets.ImageFolder(dataset_config['datadir'], dataset_config['eval_transformation'])
    ds_size = len(train_dataset.imgs)

    with open(args.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())

    labeled_idxs, unlabeled_idxs = data.relabel_dataset_dict(train_dataset, labels)

    labeled_idxs = partition_dict(labeled_idxs, outer_k)
    unlabeled_idxs = partition_dict(unlabeled_idxs, outer_k)

    # Outer cross-validation fold
    for test_idx in range(outer_k):
        train_val_idx = [i for i in range(outer_k) if i != test_idx]

        inner_fold_split = partition(train_val_idx, inner_k)

        default_params = (args.batch_size, args.labeled_batch_size / args.batch_size, args.weight_decay, args.momentum)
        # Initialize best results
        best_acc = 0
        best_params = default_params
        # select 50 random params
        hp_params = [default_params] + random.sample(hp_product(), HP_COMBINATIONS)
        for bs_hp, n_labels_ratio_hp, wd_hp, momentum_hp in hp_params:
            # update args
            args.batch_size = bs_hp
            args.labeled_batch_size = int(bs_hp * n_labels_ratio_hp)
            args.weight_decay = wd_hp
            args.momentum = momentum_hp

            # Run the inner fold
            current_accuracies = []
            for val_idx in range(inner_k):
                train_idx = [i for i in range(inner_k) if i != val_idx]

                train_idx = [j for i in train_idx for j in inner_fold_split[i]]
                val_idx = inner_fold_split[val_idx]

                train_loader = create_loader(train_dataset, labeled_idxs, unlabeled_idxs, train_idx)
                val_loader = create_loader(eval_dataset, labeled_idxs, unlabeled_idxs, val_idx, eval=True)

                results = execute_model(args, context, train_loader, val_loader)
                current_accuracies.append(results['Accuracy-top1'].val)

            # Check if the current hyper params are better
            if np.mean(current_accuracies) > best_acc:
                best_acc = np.mean(current_accuracies)
                best_params = bs_hp, n_labels_ratio_hp, wd_hp, momentum_hp

        # update args by best params
        bs_hp, n_labels_ratio_hp, wd_hp, momentum_hp = best_params
        args.batch_size = bs_hp
        args.labeled_batch_size = int(bs_hp * n_labels_ratio_hp)
        args.weight_decay = wd_hp
        args.momentum = momentum_hp

        # Run the outer fold
        train_val_loader = create_loader(train_dataset, labeled_idxs, unlabeled_idxs, train_val_idx)
        test_loader = create_loader(eval_dataset, labeled_idxs, unlabeled_idxs, [test_idx], eval=True)

        results = execute_model(args, context, train_val_loader, test_loader)
        results_dict = results.values()
        results_dict.pop('inference_time')
        results_dict['Inference Time'] = results['inference_time'].avg * 1000

        ncv_log.record(test_idx+1, {
            'Dataset Name': f'{args.dataset}_{args.n_labels}',
            'Algorithm Name': algo_short_to_name[args.model_arch],
            'Cross Validation': str(test_idx + 1),
            'Hyper parameter': str({'batch_size': bs_hp,
                                    'labels_ration': n_labels_ratio_hp,
                                    'weight_decay': wd_hp,
                                    'momentum': momentum_hp}),
            **results_dict
        })

    ncv_log.save()


def defaults(arch, dataset, n_labels, net_arch='cnn13'):
    args = {
        # architecture
        'model-arch': arch,
        'arch': net_arch,
        'model_num': 2 if arch == 'mt' else 4,

        # data
        'dataset': dataset,
        'n_labels': n_labels,
        'labels': os.path.join(os.getcwd(), 'data-local', 'labels', dataset, f'{n_labels}.txt'),

        # Technical Details
        'workers': 2,

        # optimization
        'batch-size': 100,
        'labeled-batch-size': 50,

        # optimizer
        'lr': 0.1,
        'nesterov': True,
        'weight-decay': 1e-4,

        # constraint
        'consistency_scale': 10.0,
        'consistency_rampup': 5,

        'stable_threshold': 0.4,
        'stabilization_scale': 100.0,
        'stabilization_rampup': 5,

        'logit_distance_cost': 0.01,

        'consistency': 100.0,  # mt-only

        'title': f'{arch}_{dataset}_{n_labels}l_{net_arch}',
        'epochs': 100,

        # debug
        'print_freq': 10,
        'validation_epochs': 1,
        'checkpoint_epochs': 1
    }
    return args


def run(title, n_labels, **kwargs):
    global args
    LOG.info('run title: %s', title)

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    context = RunContext(title, "{}".format(n_labels))
    fh = logging.FileHandler('{0}/log.txt'.format(context.result_dir))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)

    args = parse_dict_args(n_labels=n_labels, **kwargs)
    nested_cross_validation(context)


if __name__ == '__main__':
    model_arch, dataset, n_labels = sys.argv[1:]
    net_arch = 'cnn3' if dataset == 'mnist' else 'cnn13'
    args = defaults(model_arch, dataset, n_labels, net_arch)
    run(**args)
