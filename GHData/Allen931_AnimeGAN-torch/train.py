import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.hipify.hipify_python import str2bool

from src.AnimeGAN import AnimeGAN
from src.dataset import AnimeDataSet, ValidationSet
from multiprocessing import cpu_count


def collate_fn(batch):
    img, anime, anime_gray, anime_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        torch.stack(anime_smt_gray, 0),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Yurucamp')
    parser.add_argument('--train_dir', type=str, default='./content/dataset/train_photo')
    parser.add_argument('--anime_dir', type=str, default='./content/dataset/Yurucamp')
    parser.add_argument('--val_dir', type=str, default='./content/dataset/val')

    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--init_epochs', type=int, default=10)
    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--checkpoint_dir', type=str, default='./content/checkpoints/Yurucamp')
    parser.add_argument('--save_image_dir', type=str, default='./content/images')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_layers', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='lsgan / hinge / bce')
    parser.add_argument('--resume', type=str2bool, default=True)
    parser.add_argument('--use_sn', action='store_true')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--debug_samples', type=int, default=0)
    parser.add_argument('--lr_g', type=float, default=2e-5)
    parser.add_argument('--lr_d', type=float, default=4e-5)
    parser.add_argument('--lr_init', type=float, default=2e-4)

    parser.add_argument('--wadvg', type=float, default=300.0, help='Adversarial loss weight for Generator')
    parser.add_argument('--wadvd', type=float, default=300.0, help='Adversarial loss weight for Discriminator')

    parser.add_argument('--wadvd_real', type=float, default=1.7,
                        help='Adversarial loss weight for D of real anime images')
    parser.add_argument('--wadvd_gray', type=float, default=1.7,
                        help='Adversarial loss weight for D of gray anime images')
    parser.add_argument('--wadvd_fake', type=float, default=1.7,
                        help='Adversarial loss weight for D of generated anime images')
    parser.add_argument('--wadvd_smooth', type=float, default=1.0,
                        help='Adversarial loss weight for D of smooth anime images')

    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--wgray', type=float, default=2.5, help='Grayscale loss weight')
    parser.add_argument('--wcol', type=float, default=10.0, help='Color loss weight')
    parser.add_argument('--wtv', type=float, default=1.0, help='Total Variation loss weight')

    parser.add_argument('--d_noise', action='store_true')

    return parser.parse_args()


def check_params(args):
    train_dir = os.path.join(args.train_dir, args.dataset, '/')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f'Dataset not found {train_dir}')

    anime_dir = os.path.join(args.anime_dir, args.dataset, '/')
    if not os.path.exists(anime_dir):
        raise FileNotFoundError(f'Dataset not found {anime_dir}')

    val_dir = os.path.join(args.val_dir, args.dataset, '/')
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f'Dataset not found {val_dir}')

    save_image_dir = os.path.join('./', args.save_image_dir, '/')
    if not os.path.exists(save_image_dir):
        print(f'* {save_image_dir} does not exist, creating...')
        os.makedirs(save_image_dir)

    checkpoint_dir = os.path.join('./', args.checkpoint_dir, '/')
    if not os.path.exists(checkpoint_dir):
        print(f'* {checkpoint_dir} does not exist, creating...')
        os.makedirs(checkpoint_dir)

    assert args.gan_type in {'lsgan', 'hinge', 'bce'}, f'{args.gan_type} is not supported'


def main(args):
    check_params(args)

    print("Init models...")

    # Create DataLoader
    dataset = AnimeDataSet(args, args.train_dir)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataset = ValidationSet(args, args.val_dir)
    args.val_len = val_dataset.len_train

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    model = AnimeGAN(args, data_loader, val_loader)

    if args.resume:
        # Load G and D
        try:
            model.load()
            print("G weight loaded")
            print("D weight loaded")
        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)

    while model.epoch < args.epochs:
        model.process()

    print("Training finished")
    model.save()


if __name__ == '__main__':
    args = parse_args()
    main(args)
