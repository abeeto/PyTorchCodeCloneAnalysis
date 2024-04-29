#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse

from os.path import join
from datetime import datetime

from data_utils import DatasetBuilder
from utils import get_logger
from train import Model


def main():
    # directory for training outputs
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

    # required parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=32 * 2, type=int,
                        help="Batch size.")
    parser.add_argument("--multi_gpu", action='store_true', default=True,
                        help="If use multi-gpu training.")
    parser.add_argument("--write_summary", action='store_true', default=True,
                        help="If use tensorboard.")
    parser.add_argument("--vector_size", default=300, type=int,
                        help="Word embedding size.")
    parser.add_argument("--hidden_size", default=300, type=int,
                        help="LSTM hidden vector size.")
    parser.add_argument("--class_num", default=3, type=int,
                        help="Number of class.")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="Dropout rate.")
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="Optimizer type.")
    parser.add_argument("--factor", default=0.1, type=float,
                        help="Factor for ReduceLROnPlateau.")
    parser.add_argument("--patience", default=3, type=int,
                        help="Patience for ReduceLROnPlateau.")
    parser.add_argument("--lr", default=4e-4, type=float,
                        help="learning rate.")
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="Number of total epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="output directory for model, log file and summary.")
    parser.add_argument("--log_path", default=join(output_dir, "log.txt"), type=str,
                        help="Path to log.txt.")
    parser.add_argument("--summary_path", default=join(output_dir, "summary"), type=str,
                        help="Path to summary file.")
    parser.add_argument("--model_dir", default=join(output_dir, "model/"), type=str,
                        help="Directory for saved model.")

    parser.add_argument("--word_vectors", default="glove.6B.300d.txt", type=str,
                        help="Type of pre-trained word embedding should be used.")
    parser.add_argument("--vector_cache", default=".vector_cache", type=str,
                        help="Where to store downloaded word embeddings. (.vector_cache)")
    parser.add_argument("--data_folder", default=".data", type=str,
                        help="The folder contains SNLI dataset. (.data)")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)

        # init_process_group: call this before using other functions
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # Build model directory and logger file
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # bloack process other than -1 or 0

    if not os.path.exists(args.model_dir):
        assert args.local_rank in [-1, 0], "Only the first process can build model directory !!!"
        os.makedirs(args.model_dir)

    args.logger = get_logger(args.log_path)

    # build dataset
    args.logger.info("Loading dataset...")
    builder = DatasetBuilder(args)
    train, dev, test = builder.create_dataset()
    train_iter, dev_iter, test_iter = builder.get_iterator(train, dev, test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # build model
    args.logger.info("Building model...")
    args.vocab_size = len(builder.input_vocab)
    model = Model(args)
    model.initialize_embeddings(builder.input_vocab.vectors)

    # training
    args.logger.info("Start training !!!")
    model.fit(train_iter, dev_iter)

    # test & get report
    if args.local_rank in [-1, 0]:
        args.logger.info("Loading best mode and start testing:")
        model.load_weights(args.model_dir)
        model.get_report(test_iter, target_names=['False', 'Partly True', 'True'])


if __name__ == '__main__':
    main()