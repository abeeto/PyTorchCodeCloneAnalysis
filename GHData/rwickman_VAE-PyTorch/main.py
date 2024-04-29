import argparse

from trainer import Trainer


def main(args):
    model_trainer = Trainer(args)
    model_trainer.train()
    model_trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, 
        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, 
        help="Batch size.")
    parser.add_argument("--lr", type=float, default=8e-4, 
        help="Learning rate.")
    parser.add_argument("--load", action="store_true",
        help="Load the model.")
    

    main(parser.parse_args())