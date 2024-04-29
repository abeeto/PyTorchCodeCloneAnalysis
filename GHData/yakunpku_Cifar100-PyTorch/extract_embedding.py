import os
import random
import numpy as np
import logging
import torch
import torch.nn as nn
from config import setup_logger
from config import Config as cfg
from utils.serialization import load_checkpoint
from datasets import create_dataloader 
from models import define_net
from evaluator.evaluators import Evaluator


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="The arguments for training the classifier on CIFAR-100 dataset.")
    parser.add_argument('--checkpoint-path', type=str,
                        required=True, 
                        help="the pretrained classification model path")
    parser.add_argument('--num-classes', type=int,
                        default=100,
                        help="the number of classes in the classification dataset")
    parser.add_argument('--gpu', type=int, 
                        default=0, 
                        help="to assign the gpu to train the network")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    setup_logger('base', 'test')
    logger = logging.getLogger('base')

    device = "cuda:{}".format(args.gpu)
    checkpoint = load_checkpoint(args.checkpoint_path, logger)

    network = define_net(checkpoint['arch'], checkpoint['block_name'], args.num_classes).to(device)
    network.load_state_dict(checkpoint['state_dict'])

    test_dataloader = create_dataloader(cfg.test_image_dir, cfg.test_image_list, phase='test')

    test_embeddings = Evaluator.extract_embedding(network, device, test_dataloader)
    test_embeddings = np.vstack(test_embeddings)

    os.makedirs(cfg.embedding_dir, exist_ok=True)
    np.savetxt(os.path.join(cfg.embedding_dir, 'test_embeddings.list'), test_embeddings, fmt='%f')

if __name__ == '__main__':
    main()