import argparse
import random
import matplotlib.pyplot as plt
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, draw, unnormalize


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'], help="either `train` or `val`")
    args = parser.parse_args()

    cfg = load_config(args.cfg)

    seed = random.randint(0, 9999)
    dataloader_0 = create_dataloader(cfg['%s_csv' % args.split],
                                     batch_size=cfg.batch_size,
                                     image_size=cfg.input_size,
                                     augment=False,
                                     shuffle=True,
                                     seed=seed)
    dataloader_1 = create_dataloader(cfg['%s_csv' % args.split],
                                     batch_size=cfg.batch_size,
                                     image_size=cfg.input_size,
                                     augment=True,
                                     shuffle=True,
                                     seed=seed)
    dataiter_0 = iter(dataloader_0)
    dataiter_1 = iter(dataloader_1)

    while True:
        plt.figure(figsize=(15, 7))

        images, annos = next(dataiter_0)
        image = unnormalize(images[0])
        image = draw(image, annos[0], cfg.num_classes)
        plt.subplot(1, 2, 1)
        plt.title("w/o augmentation")
        plt.imshow(image.permute([1, 2, 0]))

        images, annos = next(dataiter_1)
        image = unnormalize(images[0])
        image = draw(image, annos[0], cfg.num_classes)
        plt.subplot(1, 2, 2)
        plt.title("w/ augmentation")
        plt.imshow(image.permute([1, 2, 0]))

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
