import argparse

parser = argparse.ArgumentParser('This is a test for package argparse!')
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rates')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--save_interval', type=int, default=10)

config = parser.parse_args()

def train(opt):
    print(opt.lr)
    print(opt.batchsize)
    print(opt.epochs)
    print(opt.save_interval)

train(config)

