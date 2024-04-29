import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from gan.gans import Gan


def main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)

    parser = Trainer.add_argparse_args(parser)
    parser = Gan.add_model_specific_args(parser)

    args = parser.parse_args()

    dict_args = vars(args)

    model = Gan(img_shape=(1, 28, 28), **dict_args)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=train_loader)


if __name__ == '__main__':
    main()
