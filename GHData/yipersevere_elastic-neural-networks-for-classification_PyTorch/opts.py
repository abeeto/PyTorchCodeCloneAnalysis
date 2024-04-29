import argparse

# General model args
parser = argparse.ArgumentParser(description='PyTorch Elastic-ResNet implementation')

parser.add_argument('--data', type=str, help="training and testing data, data=cifar10; data=cifar100, tiny_imagenet", default="tiny_imagenet")
parser.add_argument('--num_classes', type=int, help="classification number, 10 or 100", default=200)
parser.add_argument('--target_size', type=tuple, help='default target size is (224,224,3)', default=(224,224,3))
parser.add_argument('--debug', type=bool, help="set program as debug mode", default=False)
parser.add_argument('--multi_GPU', type=bool, help="multiple GPUs mode", default=False)
parser.add_argument('--epochs', type=int, help="epoch number, default 1, set 100 or 1000", default=100)
parser.add_argument('--dropout_rate', type=float, help="dropout rate, (default: 0.2)", default=0.2)
parser.add_argument('--batch_size', type=int, help="batch size for training and testing, (default: 16)", default=256)
parser.add_argument('--learning_rate', type=float, help="initial learning rate (default: 1e-3)", default=1e-2)
parser.add_argument('--pretrain_learning_rate', type=float, help="initial learning rate (default: 1e-3)", default=1e-3)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='Momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default="0", help='gpu available')
parser.add_argument('--add_intermediate_layers', type=int, 
                    help="add intermediate layers, 2: all intermediate layers; "
                                                    "1: skip early intermediate layers output;"
                                                    "0 : not any intermediate layers. (default: 0)", default=2)
parser.add_argument('--layers_weight_change', type=int, default=0, 
                    help="1 for giving different weights for different intermediate layers output classifiers, 0 for setting all weights are 1")

parser.add_argument('--model', type=str, help="model folder, like ElasticNN-ResNet50", default="Elastic_MobileNet")
parser.add_argument('--model_name', type=str, help="exact model name", default="pytorch_tiny_imagenet_4_intermediate_classifiers_Elastic_MobileNet_new_bp1")
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='Manual seed (default: 0)')
                    
parser.add_argument('--save-model', dest='save_model', action='store_true',
                    help='Only save best model (default: false)')
parser.add_argument('--pretrained_weight', type=int, help="flag to add imagenet pretrained weight; "
                                                        "1 means loading pretrained weight, 0 means not loading pretrained weight",default = 0)

parser.add_argument('--backpropagation', type=int, help="backprogation way to train model ", default = 1)

# Init Environment
args = parser.parse_args()