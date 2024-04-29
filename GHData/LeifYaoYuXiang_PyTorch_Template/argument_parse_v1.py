import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Demo Template")

    # dataset configuration
    parser.add_argument('--root_dir', type=str, default='')

    # model configuration
    parser.add_argument('--n_layer', type=int, default=2, help="the layer num of heroGraph")

    # training configuration
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4, help="the learning rate")
    parser.add_argument('--comment', type=str, default='default comment')
    args = parser.parse_args()
    return args
