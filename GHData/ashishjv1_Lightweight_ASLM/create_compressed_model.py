import os
import numpy as np
import models_cpu as models
import torch
import torch.nn as nn
from layers.svd import SVD_conv_layer
from layers.cpd import CPD3_layer
from utils.replacement_utils import (get_layer_by_name, replace_conv_layer_by_name)
from flopco import FlopCo
import copy
import argparse

parser = argparse.ArgumentParser(description="Create Model From Ranks")

parser.add_argument('--full_model', default='', type=str, required=False, help='Saved Full Model to create Partially Compressed Model(.pth)')
parser.add_argument('--save_path', default='', type=str, required=True, help='Dir to save Partial or Fully Decomposed Model')
parser.add_argument('--model_type', default='full', type=str, required=True, help='Create a partial(P) or a Full Model(F)')
parser.add_argument('--ranks_dir', default='~/ranks_PA_100K/', type=str, required=True, help='Dir to load ranks and CPD-Tensors')
# parser.add_argument('--device', default='cuda', type=str, required=False, help='device to use for decompositions')
parser.add_argument('--attr_num', default='26', type=int, required=True, help='(35)PETA or (26)PA-100K')
args = parser.parse_args()


def check_flops(check_model, temp_model, device='cpu'):
    '''
    Checks IF Flops Less than original layer
    '''
    sums = 0
    list_layers = []
    sum_list = []
    stats_compressed = FlopCo(check_model, img_size=(1, 3, 256, 128), device=device)
    stats_original = FlopCo(temp_model, img_size=(1, 3, 256, 128), device=device)
    compresseddict_flops = stats_compressed.relative_flops
    original_dict_flops = stats_original.relative_flops
    for olayers, ovalues in original_dict_flops.items():
        sums = 0
        for clayers, cvalues in compresseddict_flops.items():
            if olayers in clayers:
                sums = sums + cvalues
        if sums <= ovalues:
            list_layers.append(olayers)
            sum_list.append(sums)
    return list_layers, sum_list


def new_model_svd_layers(rank, build_model, names):
    '''
    Decompose and fit SVD Layers
    '''
    layer_rank = int(rank)
    layer_to_decompose = copy.deepcopy(get_layer_by_name(build_model, names))
    decomposed_svd = SVD_conv_layer(layer_to_decompose, rank_selection='manual',
                                    rank=layer_rank)
    replace_conv_layer_by_name(build_model, names, decomposed_svd)
    del decomposed_svd
    return build_model


def new_model_cpd_layers(Us_cp, build_model, names, device):
    '''
    Decompose And Fit CPD Layers
    '''
    layer_to_decompose = copy.deepcopy(get_layer_by_name(build_model, names))
    decomposed_cp = copy.deepcopy(CPD3_layer(layer_to_decompose.to(device),
                                             factors=Us_cp, cpd_type='cp')).to(device)
    replace_conv_layer_by_name(build_model, names, decomposed_cp)
    del decomposed_cp
    return build_model


def full_compression(names, lnames_to_compress, toload, temp_model):
    if names in lnames_to_compress:
        rank = np.load(toload, allow_pickle=True)
        new_model = new_model_svd_layers(rank, temp_model, names)
        return new_model
    else:
        Us_cp = np.load(toload, allow_pickle=True).tolist()
        new_model = new_model_cpd_layers(Us_cp, temp_model, names, device="cpu")
        return new_model


def part_compression(names, lnames_to_compress, toload, list_to_compress_SVD, build_model, device):
    if names in list_to_compress_SVD:
        rank = np.load(toload, allow_pickle=True)
        return(new_model_svd_layers(rank, build_model, names))
       
    elif names not in lnames_to_compress:
        Us_cp = np.load(toload, allow_pickle=True).tolist()
        return(new_model_cpd_layers(Us_cp, build_model, names, device="cpu"))





def main():

    model_dir = args.full_model
    model_type = args.model_type
    attr_num = args.attr_num
    device = "cpu"
    if model_type == 'full' or model_type == 'f' or model_type == 'FULL' or model_type == 'F':
        model = models.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
        temp_model = copy.deepcopy(model).to(device)
        lnames_to_compress = [module_name
                              for module_name, module in model.named_modules()
                              if isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1]
        directory = args.ranks_dir
        
        for files in os.listdir(directory):
            toload = os.path.join(directory + files)
            names1 = files.replace('Inception_ranks_module.', "")
            names = names1.replace('_0.002_grid_step_1.npy', "")
            full_model = full_compression(names, lnames_to_compress, toload, temp_model)
            new_model_dir = args.save_path
            torch.save(full_model, new_model_dir)

    elif model_type == 'partial' or model_type == 'p' or model_type == 'PARTIAL' or model_type == 'P':

        check_model = torch.load(model_dir).to(device)
        model = models.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
        temp_model = copy.deepcopy(model).to(device)
        lnames_to_compress_SVD, sums = check_flops(check_model, temp_model, device=device)
        
        lnames_to_compress = [module_name
                              for module_name, module in model.named_modules()
                              if isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1]
        
        list_to_compress_SVD = [items for items in lnames_to_compress_SVD if items in lnames_to_compress]
        
        build_model = copy.deepcopy(model).cpu()
        directory = args.ranks_dir
        for files in os.listdir(directory):
            toload = os.path.join(directory + files)
            # string = directory + '/Inception_ranks_module.'
            names1 = files.replace('Inception_ranks_module.', "")
            names = names1.replace('_0.002_grid_step_1.npy', "")
            part_model = part_compression(names, lnames_to_compress, toload, list_to_compress_SVD, build_model, device)
            if part_model != None:
                new_model_dir = args.save_path
                torch.save(part_model, new_model_dir)

    else:
        print('Please enter the compression Type as : --model_type (F) for Full or (P) for Partial')



if __name__ == '__main__':
    main()
