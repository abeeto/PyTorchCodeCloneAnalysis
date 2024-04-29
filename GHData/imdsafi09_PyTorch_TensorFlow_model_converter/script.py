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
import tensorflow as tf
import os

from layers import *
# from test import resnet101_c6

def tf_pytorch():
    input_data_pt = torch.ones(2,256,56,56) * 10
    input_data_tf = (torch.ones(2,56,56,256) * 10).numpy()

    data_dict = np.load('ResNet.npy').item()
    conv_tf = data_dict['layer1.1.conv1.weight']
    # 1*1*256*64

    model = model = resnet101_c6()
    model.load_state_dict(torch.load('ResNet.pth'))
    model.eval()
    conv_pt = model.state_dict()['layer1.1.conv1.weight']
    # 64*256*1*1

    # conv_tf_t = conv_tf.transpose((3,2,0,1))
    # ans = (conv_pt.numpy()-conv_tf_t).mean()
    # print ans   # the answer is 0.0

    img = tf.placeholder(tf.float32,[None,56,56,256],name='img')
    out = tf.nn.conv2d(img, conv_tf, [1, 1, 1, 1], padding='VALID')
    sess = tf.InteractiveSession()
    init=tf.global_variables_initializer()
    sess.run(init)
    ans_tf = sess.run([out], feed_dict={img:input_data_tf})
    print "the type of tf output is :", ans_tf[0].shape
    # print "the type of output is :", ans[0]

    ans_pt = torch.nn.functional.conv2d(input_data_pt, conv_pt, bias=None, stride=1, padding=0)
    print "the type of pt output is :", ans_tf[0].shape
    print ans_pt.size()
    ans = (ans_pt.numpy().transpose((0,2,3,1))-ans_tf[0]).mean()
    print ans   # the answer is 0.0


def test_size(npy_path):
    '''
    test the size of my tensorflow structure
    '''
    layers_param = [3, 4, 23, 3]

    # load the parameters
    data_dict = np.load(npy_path).item()
    weights = OrderedDict()
    for key in data_dict.keys():
        weights[key] = tf.Variable(data_dict[key], name=key)

    # print some weights
    print data_dict['bn1.weight'][:5]
    print data_dict['bn1.bias'][:5]
    print data_dict['bn1.running_mean'][:5]
    print data_dict['bn1.running_var'][:5]

    exit(1)

    # input data 
    img_data = torch.ones(2,224,224,6).numpy()*10
    
    # build the static graph

        # data placeholder
    img = tf.placeholder(tf.float32,[None,224,224,6],name='img')

        # the network
    conv1 = conv_layer(img, weights, stride=2, padding=3, bias=False, name='conv1')
    bn1 = bn_layer(conv1, weights, name='bn1')
    relu1 = tf.nn.relu(bn1)
    maxpool1 = max_pool(relu1, kernel_size=3, stride=2, padding=1, name='maxpool1')

    layer1_0 = bottleneck(maxpool1, weights, stride=1, name='layer1.0', downsample_flag=True)
    layer1_1 = bottleneck(layer1_0, weights, stride=1, name='layer1.1', downsample_flag=False)
    layer1_2 = bottleneck(layer1_1, weights, stride=1, name='layer1.2', downsample_flag=False)

    layer2_0 = bottleneck(layer1_2, weights, stride=2, name='layer2.0', downsample_flag=True)
    layer2_1 = bottleneck(layer2_0, weights, stride=1, name='layer2.1', downsample_flag=False)
    layer2_2 = bottleneck(layer2_1, weights, stride=1, name='layer2.2', downsample_flag=False)
    layer2_3 = bottleneck(layer2_2, weights, stride=1, name='layer2.3', downsample_flag=False)

    layer3_0 = bottleneck(layer2_3, weights, stride=2, name='layer3.0', downsample_flag=True)
    layer3_1 = bottleneck(layer3_0, weights, stride=1, name='layer3.1', downsample_flag=False)
    layer3_2 = bottleneck(layer3_1, weights, stride=1, name='layer3.2', downsample_flag=False)
    layer3_3 = bottleneck(layer3_2, weights, stride=1, name='layer3.3', downsample_flag=False)
    layer3_4 = bottleneck(layer3_3, weights, stride=1, name='layer3.4', downsample_flag=False)
    layer3_5 = bottleneck(layer3_4, weights, stride=1, name='layer3.5', downsample_flag=False)
    layer3_6 = bottleneck(layer3_5, weights, stride=1, name='layer3.6', downsample_flag=False)
    layer3_7 = bottleneck(layer3_6, weights, stride=1, name='layer3.7', downsample_flag=False)
    layer3_8 = bottleneck(layer3_7, weights, stride=1, name='layer3.8', downsample_flag=False)
    layer3_9 = bottleneck(layer3_8, weights, stride=1, name='layer3.9', downsample_flag=False)
    layer3_10 = bottleneck(layer3_9, weights, stride=1, name='layer3.10', downsample_flag=False)
    layer3_11 = bottleneck(layer3_10, weights, stride=1, name='layer3.11', downsample_flag=False)
    layer3_12 = bottleneck(layer3_11, weights, stride=1, name='layer3.12', downsample_flag=False)
    layer3_13 = bottleneck(layer3_12, weights, stride=1, name='layer3.13', downsample_flag=False)
    layer3_14 = bottleneck(layer3_13, weights, stride=1, name='layer3.14', downsample_flag=False)
    layer3_15 = bottleneck(layer3_14, weights, stride=1, name='layer3.15', downsample_flag=False)
    layer3_16 = bottleneck(layer3_15, weights, stride=1, name='layer3.16', downsample_flag=False)
    layer3_17 = bottleneck(layer3_16, weights, stride=1, name='layer3.17', downsample_flag=False)
    layer3_18 = bottleneck(layer3_17, weights, stride=1, name='layer3.18', downsample_flag=False)
    layer3_19 = bottleneck(layer3_18, weights, stride=1, name='layer3.19', downsample_flag=False)
    layer3_20 = bottleneck(layer3_19, weights, stride=1, name='layer3.20', downsample_flag=False)
    layer3_21 = bottleneck(layer3_20, weights, stride=1, name='layer3.21', downsample_flag=False)
    layer3_22 = bottleneck(layer3_21, weights, stride=1, name='layer3.22', downsample_flag=False)

    layer4_0 = bottleneck(layer3_22, weights, stride=2, name='layer4.0', downsample_flag=True)
    layer4_1 = bottleneck(layer4_0, weights, stride=1, name='layer4.1', downsample_flag=False)
    layer4_2 = bottleneck(layer4_1, weights, stride=1, name='layer4.2', downsample_flag=False)

    avg_pool  = tf.layers.average_pooling2d(layer4_2, pool_size=[7,7], strides=[1,1],name='avg_pool')
    size_oper = tf.squeeze(avg_pool)
    out = tf.nn.bias_add(tf.matmul(size_oper, weights['fc.weight']), weights['fc.bias'])
    



    
    # run the session
    sess = tf.InteractiveSession()
    init=tf.global_variables_initializer()
    model_run = sess.run(init)
    print type(model_run)

    exit(1)
    ans = sess.run([out], feed_dict={img:img_data})
    print "the type of output is :", ans[0].shape
    print "the type of output is :", ans[0]
    return ans
    


def tf_network(img, npy_path, scope):
    # load the model from npy file
    data_dict = np.load(npy_path).item()

    with tf.variable_scope(scope):
        # save all the pramaters to a dict
        weights = OrderedDict()
        for key in data_dict.keys():
            weights[key] = tf.Variable(data_dict[key], name=key)


        # build the static graph
        conv1 = conv_layer(img, weights, stride=2, padding=3, bias=False, name='conv1')
        bn1 = bn_layer(conv1, data_dict, name='bn1')
        relu1 = tf.nn.relu(bn1)
        maxpool1 = max_pool(relu1, kernel_size=3, stride=2, padding=1, name='maxpool1')



def list2json(out_list, addr):
    # write a list to json file
    with open(addr, 'w+') as jfile:
        json.dump(out_list, jfile, indent=4)


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

def load_model(model_path, model_type, save_flag=False, save_path = 'ResNet.pth'):
    '''
    support loading model from all types of saving methods
    save the result model to ResNet.pth
    return a model dict
    '''
    assert model_type in ['whole_model','model_dict','whole_dict'], "Wrong model type!"
    if model_type == 'whole_model':
        model = torch.load(model_path)
        state_dict = model.state_dict()
    elif model_type == 'model_dict':
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path)['state_dict']

    # remove module if having one
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
    # npy2pytorch('ResNet.npy','ResNet_new.pth')
    # model_name = 'ResNet.pth'
    # state_dict = load_model(model_name, 'model_dict')
    # print type(state_dict['conv1.weight'])
    # # check_keys_value_size(state_dict)
    # save_as_npy(state_dict)

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf_pytorch()
    test_size('ResNet.npy')