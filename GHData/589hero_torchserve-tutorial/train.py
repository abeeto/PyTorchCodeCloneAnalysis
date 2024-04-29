import argparse
from argparse import Namespace

from trainer import MNISTTrainer
from model import MNISTClassifier
from utils import get_device, get_mnist_dataloader, get_optimizer, get_criterion


def define_arg_parser() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="The path where data is stored"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default='./model/mnist_model.pth',
        help="The filename of the model to save"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for the random number generator"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="The learning rate for training.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='adam',
        help="The optimizer to use for training.",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default='crossentropy',
        help="The loss function to use for training.",
    )
    
    return parser.parse_args()


def main():
    config = define_arg_parser()
    
    train_loader = get_mnist_dataloader(config, train=True)
    val_loader = get_mnist_dataloader(config, train=False)

    model = MNISTClassifier()

    criterion = get_criterion(config.criterion)
    optimizer = get_optimizer(config.optimizer)(model.parameters(), lr=config.learning_rate)

    trainer = MNISTTrainer(config, model, train_loader, val_loader, criterion, optimizer)
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
