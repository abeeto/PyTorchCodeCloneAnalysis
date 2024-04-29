import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description='Utility for training diffusion models.')

    parser.add_argument(
        'data_dir',
        metavar='data_dirs',
        nargs='+',
        type=str,
        help='Path to dataset roots.'
    )
    parser.add_argument(
        '-cfg', '--config',
        type=str,
        help='Run configuration.',
        default=None
    )
    parser.add_argument(
        '--continue_ckpt',
        type=str,
        help='Checkpoint to load.',
        default=None
    )

    return parser.parse_args()


def get_sample_args():
    parser = argparse.ArgumentParser(description='Utility for sampling from diffusion models.')

    parser.add_argument(
        'dst_dir',
        metavar='dst_dirs',
        type=str,
        help='Path to destination root.'
    )
    parser.add_argument(
        '-ckpt', '--checkpoint',
        type=str,
        help='Checkpoint.'
    )
    parser.add_argument(
        '-n_cls', '--n_images_per_class',
        type=int,
        help='Number of images per class.'
    )

    return parser.parse_args()
