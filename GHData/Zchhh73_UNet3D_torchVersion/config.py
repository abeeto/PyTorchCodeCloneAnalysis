import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path', default='/hdd/chenkecheng/zchhh_data/3Dtrain/fixed_data/train',
                    help='fixed trainset root path')

parser.add_argument('--save', default='UNet3D', help='save path of trained model')

parser.add_argument('--resize_scale', type=float, default=0.5, help='resize scale for input data')

parser.add_argument('--crop_size', type=list, default=[16, 96, 96], help='patch size of train samples after resize')

parser.add_argument('--batch_size', type=list, default=2, help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--early-stop', default=10, type=int, help='early stopping (default: 20)')
args = parser.parse_args()
