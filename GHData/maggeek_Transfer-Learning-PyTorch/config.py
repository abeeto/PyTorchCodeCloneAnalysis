import argparse
import torchvision.models as models
import torchvision.datasets as datasets
import datetime
import os

# arguments
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
dataset_names = ["MNIST", "CIFAR10", "CIFAR100"]
# "FashionMNIST", "LSUN", "ImageNet"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--b1', type=float, default=0.50, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.997, help='adam: decay of second order momentum of gradient')
parser.add_argument('--lr', type=float, default=0.00001, help='adam: learning rate')
parser.add_argument('--momentum', type=float, default=0.8, help='batch normalisation momentum')
parser.add_argument('--batch_size', default=64, type=int, help='number of images in a batch')
parser.add_argument('--img_size', type=int, default=224, help='height and width of resized images')
parser.add_argument('--n_epochs', default=25, type=int, help='number of epochs of training')
parser.add_argument('--n_channels', default=3, type=int, help='number of image channels')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--arch', '-a', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--data', '-d', default=datasets.CIFAR10, choices=dataset_names,
                    help='dataset: ' + ' | '.join(dataset_names) + ' (default: datasets.CIFAR10)')
parser.add_argument('--pretrained', default=False, type=bool, help='use pre-trained model')
parser.add_argument('--folder_name', type=str, default='results2', help='name of the folder to save the files')

#parser.add_argument('--sample_interval', type=int, default=200, help='interval between image saves')

args = parser.parse_args()

if not os.path.exists(args.folder_name):
    os.makedirs(args.folder_name)

# open text file with append mode
#file = open('models.txt', 'a')
#file.write(str(datetime.datetime.now()) + '\n' + str(opt) + '\n')
#file.close()