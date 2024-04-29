import os
import sys
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='PyTorch EBM Args')
    parser.add_argument('--env', default="Widow250PickTray-v0",
                        help='Roboverse environment (default: Widow250PickTray-v0)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--steps', type=int, default=1000000, metavar='G',
                        help='training steps (default: 1000000)')
    parser.add_argument('--vae_epochs', type=int, default=20000, metavar='G',
                        help='VAE training epochs (default: 20000)')
    parser.add_argument('--ebm_epochs', type=int, default=2000, metavar='G',
                        help='EBM training epochs (default: 2000)')
    parser.add_argument('--lr_ebm', type=float, default=0.0001, metavar='G',
                        help='ebm lr (default: 0.0001)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                        help='lr (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='G',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--decay_epochs', type=list, default=[60, 80, 140, 280], metavar='G',
                        help='decay_epochs (default: [160, 180])')
    parser.add_argument('--decay_rate', type=float, default=0.5, metavar='G',
                        help='decay rate (default: 0.3)')
    parser.add_argument('--p_weight', type=float, default=1, metavar='G',
                        help='p weight (default: 1)')
    parser.add_argument('--image_augmentation', type=bool, default=False, metavar='G',
                        help='image augmentation (default: False)')
    parser.add_argument('--reinit_freq', type=float, default=0.05, metavar='G',
                        help='reinitialize sample weight (default: 0.05)')
    parser.add_argument('--clip', type=float, default=200, metavar='G',
                        help='max gradient norm (default: 200)')
    parser.add_argument('--sgld_steps', type=int, default=40, metavar='G',
                        help='sgld steps (default: 20)')
    parser.add_argument('--lr_sgld', type=float, default=1, metavar='G',
                        help='sgld lr (default: 1)')
    parser.add_argument('--sgld_std', type=float, default=0.01, metavar='G',
                        help='sgld std (default: 0.01)')
    parser.add_argument('--noise_std', type=float, default=0.01, metavar='G',
                        help='noise std (default: 0.01)')
    parser.add_argument('--score_loss', type=str, default='dsm', metavar='G',
                        help='ncsn score loss (default: dsm)')
    parser.add_argument('--seed', type=int, default=0, metavar='G',
                        help='random seed (default: 0)')
    parser.add_argument('--tqdm', type=bool, default=True, metavar='G',
                        help='Use tqdm progress bar (default: True)')
    parser.add_argument('--save_dir', type=str, default='./logs', metavar='G',
                        help='Directory for logs (default: ./logs)')
    parser.add_argument('--log_freq', type=int, default=10, metavar='G',
                        help='logging interval (default: 10)')
    parser.add_argument('--load_model', type=bool, default=False, metavar='G',
                        help='Load pretrained model (default: False)')
    parser.add_argument('--use_positive_rew', type=bool, default=True, metavar='G',
                        help='shape dense rewards (default: True)')
    parser.add_argument('--buffer', type=str, default='./data/pickplace_task.npy', metavar='G',
                        help='Directory for dataset (default: ./data/pickplace_task.npy)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='G',
                        help='batch size (default: 256)')

    return parser.parse_args()
