import os
import random
import time

import torch
import numpy as np
import torch.nn as nn
from importlib import import_module
from torch.utils.data import DataLoader

from options import args
from train_test import train, test
from utils import my_collate_fn, MyDataset, build_dataset, early_stop, get_vocab, print_metrics, save_results

if __name__ == '__main__':
    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # Load vocab
    vocab = get_vocab(args)

    # Load Data
    train_data = build_dataset(args, args.train_path, vocab)
    dev_data = build_dataset(args, args.dev_path, vocab)
    test_data = build_dataset(args, args.test_path, vocab)

    train_loader = DataLoader(MyDataset(train_data), args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    dev_loader = DataLoader(MyDataset(dev_data), args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_loader = DataLoader(MyDataset(test_data), args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    test_only = False
    # Define Model
    model = import_module('models.' + args.model_name).Model(args, vocab)

    if args.gpu >= 0:
        model = model.cuda(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_scores = []
    for epoch in range(args.epoch_nums):
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model_name, time.strftime('%b_%d_%H_%M_%S', time.localtime()),
                                                               'seed', str(args.random_seed)]))
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))
        if not test_only:
            # train
            epoch_start = time.time()
            losses = train(model, args, epoch, train_loader, optimizer)
            epoch_end = time.time()
            print("Train finish in %.2fs, loss is %.4f" % (epoch_end - epoch_start, np.mean(losses)))

        # dev
        dev_start = time.time()
        dev_metrics, dev_report, losses_dev = test(model, args, dev_loader)
        dev_end = time.time()
        if test_only:
            print('======== The Best Result on Dev ========')
        print("Development finish %.2fs, loss is %.4f" % (dev_end - dev_start, np.mean(losses_dev)))
        print_metrics(dev_metrics)
        best_scores.append(dev_metrics[args.criterion])
        save_results(args, model, model_dir, np.array(best_scores))

        # test
        if test_only or epoch == args.epoch_nums - 1:
            test_metrics, test_report, losses_test = test(model, args, test_loader)
            print('======== The Best Result on Test ========')
            print_metrics(test_metrics)
            print("the test report is \n", test_report)

        if test_only:
            break

        if early_stop(np.array(best_scores), args.criterion, args.patience):
            print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
            test_only = True
            args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
            model = import_module('models.' + args.model_name).Model(args, vocab)

            if args.test_model:
                sd = torch.load(args.test_model)
                model.load_state_dict(sd)

            if args.gpu >= 0:
                model.cuda(args.gpu)

