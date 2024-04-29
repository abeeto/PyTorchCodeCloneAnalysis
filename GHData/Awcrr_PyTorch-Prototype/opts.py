import argparse
import models
import datasets

parser = argparse.ArgumentParser(description='Parser for all the training options')

dataset_choices = sorted(name[:-3] for name in datasets.__dict__
        if name.endswith('Set'))
model_choices = sorted(name[:-5] for name in models.__dict__
        if name.endswith('Model'))

# General options
parser.add_argument('-data_dir', required=True, help='Path to data directory')
parser.add_argument('-dataset', required=True, choices=dataset_choices)
parser.add_argument('-batch_size', default=128, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-shuffle', default=True, action='store_false', help='Reshuffle data at each epoch')
parser.add_argument('-sampler', default=None, help='Strategy to draw examples from dataset')
parser.add_argument('-workers', default=4, type=int, help='Number of subprocesses to to load data')
parser.add_argument('-test_only', default=False, action='store_true', help='Only conduct testing on the validation set')
parser.add_argument('-training_record', default=None, help='Path to save training record, default format [[train_top1, train_top5, train_loss, val_top1, val_top5]]')

# Training options 
parser.add_argument('-save_path', default='checkpoints', help='Path to save training record')
parser.add_argument('-lr', default=0.1, type=float, help='Base learning rate of training')
parser.add_argument('-momentum', default=0.9, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=1e-4, type=float, help='Weight decay for training')
parser.add_argument('-n_epochs', default=90, type=int, help='Training epochs')

# Model options
parser.add_argument('-pretrained', default=None, help='Path to the pretrained model')
parser.add_argument('-resume', default=None, help='Path to resume training from a previous checkpoint')
parser.add_argument('-model', default='LeNet', choices=model_choices, help='Model type when we create a new one')
parser.add_argument('-nGPU', default=1, type=int, help='Number of GPUs for training')
parser.add_argument('-criterion', default='CrossEntropy', help='Type of objective function')
parser.add_argument('-num_classes', default=100, type=int, help='Num of classes in the classfication task')

args = parser.parse_args()
