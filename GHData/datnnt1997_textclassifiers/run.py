import os
import argparse
from cores import train, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'test', 'predict'])
    parser.add_argument("--config", default=None, type=str, required=True, help="The config file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError("Config file ({}) not exists or is empty !!!".format(args.config))
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'test':
        raise NotImplementedError
    elif args.mode == 'predict':
        predict(args.config)
    else:
        raise ValueError("Mode ({}) not exists or is empty !!!".format(args.mode))

