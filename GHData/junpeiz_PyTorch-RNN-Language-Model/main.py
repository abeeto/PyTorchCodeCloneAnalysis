import os
import torch
import numpy as np
import random

from opts import get_opts
from trainer import train, generate_text
from prepro import get_data


def main():
    opt = get_opts()
    opt.use_cuda = torch.cuda.is_available()

    # Set seed for reproducibility
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    corpus = get_data(opt)
    if opt.mode == 'train':
        train(opt, corpus)

    if opt.mode == 'generate':
        generate_text(opt, corpus)


if __name__ == '__main__':
    main()
