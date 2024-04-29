import numpy as np
import torch
import torch.nn as nn
import torchvision




# Recursive function to go down all submodule rabbitholes that make up a pretrained NN
def grab_layers(my_list, nn):

    if len(list(nn.children()))>0:
        for i in range(len(list(nn.children()))):
            my_list = grab_layers(my_list, list(nn.children())[i])
    else:
        my_list.append(nn)

    return my_list





def get_resnet18_extractor(param_list):

    ################
    # Import model #
    ################
    res=torchvision.models.resnet18(pretrained=True) 
    modules=list(res.children())[:-2]   # Get  right number of layers
    res=nn.Sequential(*modules) 

    #print(res)
    
    for p in res.parameters():
            p.requires_grad = True
            param_list.append(p)

    drop_layers=[];
    out_dim_list=[];

    # Do a pseudo-forward pass to find downsampling layers
    x=torch.randn(1, 3, 512, 512)
    for i in range(len(res)):
        y=res[i](x)
        if x.size()[-1]!=y.size()[-1]:
            drop_layers.append(i-1)
            out_dim_list.append(x.size()[1])    
            #print(i-1, x.size()[1])
        x=y
    out_dim_list.append(x.size()[1])


    ###################################################
    ################# FREEZE BN STATS #################
    ###################################################
    my_list=grab_layers([], res)
    print(len(my_list))
    for block in my_list:
        if isinstance(block, torch.nn.modules.BatchNorm2d):
            block.eval()
            block.track_running_stats=False
    ###################################################
    ###################################################
    ###################################################


    return res, param_list, drop_layers, out_dim_list


'''
res, param_list, drop_layers, out_dim_list = get_resnet34_extractor([])

num_scales_aux=3;
scales_4_inf=drop_layers[-num_scales_aux+1:]

print('outdim list')
print(out_dim_list)
print('drop layers')
print(drop_layers)
print('scales for inf')
print(scales_4_inf)
'''




