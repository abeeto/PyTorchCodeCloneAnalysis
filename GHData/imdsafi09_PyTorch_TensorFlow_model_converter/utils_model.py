# Copyright 2019 Chufan Wu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This doc aims to transfer PyTorch ResNet 101 model to Tensorflow model

'''

import torch
import json
from collections import OrderedDict
import numpy as np
import os

from layers import *


def pth2npy(npy_path,pth_path):
    '''
    TensorFlow 4D Tensor is NHWC format while PyTorch is NCHW format
    convert the weight shape to corrsponding TensorFlow form
    '''
    state_dict = load_model(pth_path, 'model_dict')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_v = v.numpy()
        try:
            _tmp = new_v[0,0,0,0]
            new_v = new_v.transpose((2,3,1,0))
        except IndexError:
            pass
        new_state_dict[k] = new_v
    new_state_dict['fc.weight']=new_state_dict['fc.weight'].transpose((1,0))
    np.save(npy_path, new_state_dict)
    return new_state_dict

def npy2pth(npy_path,pth_path):
    state_dict = np.load(npy_path).item()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        try:
            _tmp = v[0,0,0,0]
            new_v = v.transpose((3,2,0,1))
        except IndexError:
            new_v = v
        new_v = torch.tensor(new_v)
        new_state_dict[k] = new_v
    new_state_dict['fc.weight']=new_state_dict['fc.weight'].t()
    torch.save(new_state_dict, pth_path)
    return new_state_dict



def list2json(out_list, addr):
    # write a list to json file
    with open(addr, 'w+') as jfile:
        json.dump(out_list, jfile, indent=4)


def load_model(model_path):
    '''
    load PyTorch model from .pth file
    model can be saved as entire model or state_dict
    '''
    model = torch.load(model_path)
    try:
        # load as whole_model
        state_dict = model.state_dict()
    except:
        # load only state_dict
        state_dict = model

    # remove module text if having one
    # convert all weights to CPU
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        pos_module = k.find('module')
        if pos_module != -1:
            name = k[k.find('.')+1:]
        new_state_dict[name] = v.cpu()

    if save_flag:   # save according to model_dict
        torch.save(new_state_dict, save_path)
    return new_state_dict

def check_keys_value_size(model_state_dict, save_json_name='model_keys.json'):
    '''
    input a model dict, return a written json file
    check the keys and corrsponding sizes in a model state dict
    output a json file for model structure
    '''
    model_keys = []
    for key in model_state_dict.keys():
        model_keys.append((key, model_state_dict[key].size()))

    list2json(model_keys, save_json_name)





if __name__ == '__main__':
    npy2pth('ResNet.npy','ResNet_new.pth')
