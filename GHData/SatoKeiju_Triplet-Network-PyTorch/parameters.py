import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--save-model', action='store_true', default=False)

args = parser.parse_args()
