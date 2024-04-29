import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import TestSet

from src.networks import Generator
from utils import denormalize_input
from utils.adjust_brightness import adjust_brightness_from_src_to_dst


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--checkpoint_dir', type=str, default='./content/checkpoints',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='./content/dataset/test/HR_photo',
                        help='Directory name of test photos')
    parser.add_argument('--dataset', type=str, default='Yurucamp',
                        help='what style you want to get')
    parser.add_argument('--if_adjust_brightness', type=bool, default=True,
                        help='adjust brightness by the real photo')
    parser.add_argument('--save_image_dir', type=str, default='./content/test_result')

    return parser.parse_args()


def check_params(args):
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, 'gen')
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f'Checkpoint directory not found {checkpoint_dir}')

    save_image_dir = os.path.join(args.save_image_dir)
    if not os.path.exists(save_image_dir):
        print(f'* {save_image_dir} does not exist, creating...')
        os.makedirs(save_image_dir)

    test_dir = os.path.join(args.test_dir)
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f'Directory not found {test_dir}')


def main(args):
    check_params(args)

    dataset = TestSet(args)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    generator = Generator()
    if torch.cuda.is_available():
        generator = generator.cuda()
        if len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')) > 1:
            print(' Using %d GPU(s)' % len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')))
            generator = nn.DataParallel(generator)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, 'gen')
    if os.path.exists(checkpoint_dir):
        print('Loading %s generator...' % args.dataset)
        files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if
                 f.endswith('.pth')]
        file = max(files, key=lambda f: int(f.split('_')[-3]) * 1000000 + int(f.split('_')[-2]))

        if torch.cuda.is_available():
            data = torch.load(file)
        else:
            data = torch.load(file, map_location=lambda storage, loc: storage)

        generator.load_state_dict(data['generator'])
        print("Generator loaded")

    generator.eval()

    max_iter = len(data_loader)

    fake_imgs = []

    for index, img in enumerate(data_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
            fake_img = generator(img)
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img = fake_img.transpose(0, 2, 3, 1).squeeze(0)
            fake_img = denormalize_input(fake_img, dtype=np.int16)
            fake_imgs.append(fake_img)

            print(f'Test: {index + 1}/{max_iter}')

    save_path = os.path.join(args.save_image_dir, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, img in enumerate(fake_imgs):
        save_file = os.path.join(save_path, f'{i:03d}.png')
        print(f'* Saving {save_file}')
        cv2.imwrite(save_file, cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    args = parse_args()
    main(args)