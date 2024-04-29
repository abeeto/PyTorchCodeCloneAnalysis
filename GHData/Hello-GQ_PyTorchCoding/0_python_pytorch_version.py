# -*- coding: UTF-8 -*-
import torch
import platform


def main():
    # python version
    print('python version:', platform.python_version())

    # pytorch version
    print('pytorch version:', torch.__version__)

    # cuda is available ?
    print('cuda:', torch.cuda.is_available())


if __name__ == '__main__':
    main()
