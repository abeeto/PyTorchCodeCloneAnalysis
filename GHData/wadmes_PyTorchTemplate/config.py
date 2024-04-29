# Used for the configuration of the model
import torch
import warnings
import argparse
import sys
import torch.nn as nn

class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = argparse.ArgumentParser("Template Model")

# How the bool argument is set
parser.add_argument('--train', action='store_true', 
                    help="train the model or test the model")
parser.add_argument("--cuda", action="store_true", 
                    help="use cuda to train the model?")

# How the str argument is set
parser.add_argument("--loss", type=str, default="cross_entropy",
                    help="loss function calculate, can be 'cross_entropy' or 'weighted_cross_entropy' or 'focal_loss' ")
parser.add_argument("--name", type=str, default="template",
                    help="model name")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="load checkpint path, should be specified when test is enabled")

# How the int argument is set 
parser.add_argument("--batch_size", type=int, default=64, 
                    help="training batch size")
parser.add_argument("--num_epoch", type=int,
                    default=400, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning Rate. default=0.001")
parser.add_argument("--save_freq", type=int, default=10,
                    help="save checkpoint every SAVE_FREQ epoches")
parser.add_argument("--gpu", type=int, default=0,
                    help="gpu index")                   

opt = parser.parse_args()
if(opt.train == False):
    assert(opt.checkpoint != None)
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)