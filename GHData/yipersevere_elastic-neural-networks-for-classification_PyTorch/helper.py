from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams["figure.figsize"] = (11,5)

import os
import numpy as np
import scipy
import datetime
from io import StringIO
import shutil
import random
import sys
import time
import sys
import csv


import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


count_ops = 0
count_params = 0
module_number = 0
modules_flops = []
modules_params = []
to_print = False




label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def accuracy(output, target):
#     """Computes the precision@k for the specified values of k"""
#     # maxk = max(topk)
#     batch_size = target.size(0)
    
#     _, predicted = torch.max(output.data, 1)
#     correct += (predicted == target).sum().item()
    
#     # _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
    
#     return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def LOG(message, logFile):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)

def log_summary(model, logFile):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.summary()

    sys.stdout = old_stdout
    summary = mystdout.getvalue()

    LOG("Model summary:", logFile)

    for line in summary.split("\n"):
        LOG(line, logFile) 


def log_stats(path, epochs_acc_train, epochs_loss_train, epochs_lr, epochs_acc_test, epochs_loss_test, accs_top5, test_top5_accs):

    with open(path + os.sep + "train_errors.txt", "a") as fp:
        for a in epochs_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "train_losses.txt", "a") as fp:
        for loss in epochs_loss_train:
            fp.write("%.4f " % loss)
        fp.write("\n")

    with open(path + os.sep + "epochs_lr.txt", "a") as fp:
        fp.write("%.7f " % epochs_lr)
        fp.write("\n")    

    with open(path + os.sep + "test_errors.txt", "a") as fp:
        for a in epochs_acc_test:
            fp.write("%.4f " % a)
        fp.write("\n")
    
    with open(path + os.sep + "test_losses.txt", "a") as fp:
        for loss in epochs_loss_test:
            fp.write("%.4f " % loss)
        fp.write("\n")

    with open(path + os.sep + "train_top5_errors.txt", "a") as fp:
        for a in accs_top5:
            fp.write("%.4f " % a)
        fp.write("\n")
    
    with open(path + os.sep + "test_top5_errors.txt", "a") as fp:
        for loss in test_top5_accs:
            fp.write("%.4f " % loss)
        fp.write("\n")
    
    
def plot_figs(epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses, args, captionStrDict):
    
    """
    plot epoch test error after model testing is finished
    """

    all_y_labels = ["train error (%)", "train loss", "test error (%)", "test loss"]
    save_file_names = ["train_error.png","train_loss.png","test_error.png","test_loss.png"]
    fig_titles = [args.model + " Train Classification error"+captionStrDict["fig_title"], args.model + " Train Loss"+captionStrDict["fig_title"], args.model + " Test Classification error"+captionStrDict["fig_title"], args.model + " Test Loss"+captionStrDict["fig_title"]]
    all_stats = [epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses]
    for y_label, file_name, fig_title, data in zip(all_y_labels, save_file_names, fig_titles, all_stats):

        fig, ax0 = plt.subplots(1, sharex=True)
        colormap = plt.cm.tab20

        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(data[0]))])

        last = len(data[0])-1

        for k in range(len(data[0])):
            # Plots
            x = np.arange(len(data)) + 1
            y = np.array(data)[:, k]

            if y_label in ["train loss", "test loss"] and len(data[0]) > 1: # means model generates more than one classifier
                if k == last:
                    c_label = "total sum loss"
                elif k == (last-1):
                    c_label = captionStrDict["elastic_final_layer_label"]
                else:
                    c_label = captionStrDict["elastic_intermediate_layer_label"] + str(k)                

            else:
                if k == last:
                    c_label = captionStrDict["elastic_final_layer_label"]
                else:
                    c_label = captionStrDict["elastic_intermediate_layer_label"] + str(k)

            ax0.plot(x, y, label=c_label)
        
        ax0.set_ylabel(y_label)
        ax0.set_xlabel(captionStrDict["x_label"])
        ax0.set_title(fig_title)

        ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig_size = plt.rcParams["figure.figsize"]

        plt.rcParams["figure.figsize"] = fig_size
        plt.tight_layout()

        plt.savefig(args.savedir + os.sep + file_name)
        plt.close("all")  

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params, module_number, modules_flops
    global modules_params, to_print
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    if to_print:
        print("")

    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        if hasattr(layer, 'shared'):
            delta_params = delta_params / int(layer.shared)
        module_number += 1
        modules_flops.append(delta_ops)
        modules_params.append(delta_params)
        if to_print:
            print(layer)
            print("Module number: ", module_number)
            print("FLOPS:", delta_ops)
            print("Parameter:", delta_params)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)
        # module_number += 1
        # modules_flops.append(delta_ops)
        # to_print:
        #   print(layer)
        #   print("Module number: ", module_number)
        #   print("FLOPS:", delta_ops)


    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)
        # module_number += 1
        # modules_flops.append(delta_ops)
        # if to_print:
        #   print("Module number: ", module_number)
        #   print("FLOPS:", delta_ops)
        #   print("##Current params: ", count_params)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    layer.flops = delta_ops
    layer.params = delta_params
    return


def measure_model(model, H, W, debug=False):
    global count_ops, count_params, module_number, modules_flops
    global modules_params, to_print
    count_ops = 0
    count_params = 0
    module_number = 0
    modules_flops = []
    modules_params = []
    to_print = debug

    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    if to_print:
        print("modules flops sum: ", sum(modules_flops[0:2]))
    return count_ops, count_params

def save_checkpoint(state, args):
    
    model_dir = os.path.join(args.savedir, 'save_models')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    torch.save(state, best_filename)
    print("=> saved checkpoint '{}'".format(best_filename))

    return

def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):

    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr