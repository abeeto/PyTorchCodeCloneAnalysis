
# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def freeze_first_n_layers(model, numb_layers):
    '''
    Freezing the first numb_layers of a model
    :param model:
    :param numb_layers:
    :return:
    '''
    for name, child in model.named_children():
        numb_layers -= 1
        for name2, params in child.named_parameters():
            params.requires_grad = (numb_layers <= 0)

        return model

def print_layers(model):
    '''
    print out model layers
    :param model:
    :param numb_layers:
    :return:
    '''
    for name, child in model.named_children():
        print("name:" + str(name) + " child:" + str(child))
        # for name2, params in child.named_parameters():
        #     print("name2:" + name2 + " params:" + str(params))

def print_layers1(model):
    '''
    print out model layers
    :param model:
    :param numb_layers:
    :return:
    '''
    for index, module in enumerate(model.modules()):
        print("index:" + str(index) + " module:" + str(module.parameters))
        # for name2, params in child.named_parameters():
        #     print("name2:" + name2 + " params:" + str(params))


def print_optimizer_params(opt):
    for g in opt.param_groups:
        print(g['lr'])

#load pretrained model
model_conv = torchvision.models.resnet50(pretrained=True)

lr= .001
lr_mult = 0.33

ml = list()
ml.append({'params': model_conv.layer1.parameters(), 'lr': lr*lr_mult*lr_mult})
ml.append({'params': model_conv.layer2.parameters(), 'lr': lr*lr_mult*lr_mult})
ml.append({'params': model_conv.layer3.parameters(), 'lr': lr*lr_mult})
ml.append({'params': model_conv.layer4.parameters(), 'lr': lr*lr_mult})
ml.append({'params': model_conv.fc.parameters(), 'lr': lr})

optimizer_conv= optim.Adam(ml, lr=.001, eps=1e-8, weight_decay=1e-5)
print_optimizer_params(optimizer_conv)
print("done1")

##############################################
def get_param_list(model, lr):
    '''
    apply learning rate list evenly to model layers

    :param model:
    :param lr: list[the last is the fc layer learning rate, the rest are divided amongst the layers
    :return:
    '''
    numb_layers = 0
    for name, module in model_conv.named_children():
        if ('layer') in name:
            numb_layers += 1

    # get num layers per learning rate
    nlplr = numb_layers // (len(lr) - 1)

    params_dict = dict(model_conv.named_parameters())
    params = []
    for key, value in params_dict.items():
        print(key)
        if key[:len("layer")] == "layer":
            # probably looks like layer1.0.xxxxx
            #get the layer1 bit

            layer_number = int(key[len("layer"):].split('.')[0])
            params += [{'params': [value], 'lr': lr[layer_number // nlplr]}]
        elif key[:len("fc")] == "fc":
            params += [{'params': [value], 'lr': lr[len(lr)-1]}]
    return params
#####################################################

lrs = [.001, .01]
params = get_param_list(model_conv, lrs)
op = optim.Adam(params, lr=.009, eps=1e-8, weight_decay=1e-5)

for g in op.param_groups:
    print (g['lr'])
    # print(g)
print("done2")





#another way to view params
# params_dict = dict(model_conv.named_parameters())
# print (params_dict)

# opt.lr= 10
# optimizer = optim.Adam([{'params': model_conv.modules[14].parameters(), 'lr': 0.1*opt.lr}, {'params': [module.parameters() for index, module in enumerate(model_conv.modules) if index != 14]}], lr=opt.lr, weight_decay=1e-5)

# print_layers(model_conv)
# print(model_conv.layer1)

# print(model_conv.modules())


# print(model_conv.modules())

# for module in model_conv.modules():
#     print(module)
# for module in model_conv.modules():
#     print(module.parameters.lr)


    # ml = list()
    # for i in range(layers):
    #     ml.append({'params': model.["layer1"].parameters(), 'lr':

    # for name, module in model_conv.named_children():
    #     if ('layer') in name:
    #         # get the number of the layer
    #         n = int(name[len("layer"):])
    #
    #




        # print_layers1(model_conv)
        #
        # ml = list()
        # ml.append({'params': model_conv.modules[3].parameters(), 'lr': .001})
        # for index, module in enumerate(model_conv.modules):
        #     if (index != 3):
        #         ml.append({'params': module.parameters()})
        # optimizer = optim.Adam(ml, lr=.01, eps=1e-8, weight_decay=1e-5)




        # #freeze all but laast layer
        # for param in model_conv.parameters():
        #     param.requires_grad = False
        #
        # num_linear_inputs = model_conv.fc.in_features
        # num_outputs = 12 # number of weedlings
        #
        # model_conv.fc = nn.Linear(num_linear_inputs, num_outputs)
        #
        # model_conv = model_conv.to(device)
        #
        # criterion = nn.CrossEntropyLoss()
        #
        # # Observe that only parameters of final layer are being optimized as
        # # opoosed to before.

        # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        # for g in optimizer_conv.param_groups:
        #     print (g['lr'])
        #     print(g)

        # g['lr'] = 0.001
        #
        # # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        #
        # # train just fc layer
        # model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
        #                        num_epochs=10)
        #
        # #now train the model with last 20 layers plus fc layer
        # model_conv = freeze_first_n_layers(model_conv, 20)
        # model_conv = model_conv.to(device)
        #
        # # fine tune
        # model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
        #                        num_epochs=10)
        #
        #
        # # unfreeze
        #
        # # train again
        #
        # # final submission


# #learning rates, the last is the fully connected learning rate, the rest are applied from the end back
# lr = [.0001,.001, .01]
#
# #how many layers in model?
# numb_layers = 0
# for name, module in model_conv.named_children():
#     if ('layer') in name:
#         numb_layers+=1
#     # print(name)  # to see the names of the layers
#
# #get learning rates per layer
# layers = numb_layers//len(lr)
#
# ml = list()
# for i in range(layers):
#     ml.append({'params': model_conv.layer1.parameters(), 'lr':





# print_layers1(model_conv)
#
# ml = list()
# ml.append({'params': model_conv.modules[3].parameters(), 'lr': .001})
# for index, module in enumerate(model_conv.modules):
#     if (index != 3):
#         ml.append({'params': module.parameters()})
# optimizer = optim.Adam(ml, lr=.01, eps=1e-8, weight_decay=1e-5)




# #freeze all but laast layer
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# num_linear_inputs = model_conv.fc.in_features
# num_outputs = 12 # number of weedlings
#
# model_conv.fc = nn.Linear(num_linear_inputs, num_outputs)
#
# model_conv = model_conv.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that only parameters of final layer are being optimized as
# # opoosed to before.

# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# for g in optimizer_conv.param_groups:
#     print (g['lr'])
#     print(g)

    # g['lr'] = 0.001
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#
# # train just fc layer
# model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
#                        num_epochs=10)
#
# #now train the model with last 20 layers plus fc layer
# model_conv = freeze_first_n_layers(model_conv, 20)
# model_conv = model_conv.to(device)
#
# # fine tune
# model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
#                        num_epochs=10)
#
#
# # unfreeze
#
# # train again
#
# # final submission