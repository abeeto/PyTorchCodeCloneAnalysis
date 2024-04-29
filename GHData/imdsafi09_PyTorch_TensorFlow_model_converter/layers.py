# layers.py
import tensorflow as tf

'''
this file defines some basic layers for ResNet modules
'''


def padding_param(pad):
    # implemention for tensorflow padding mimicing pytorch
    return tf.constant([[0,0],[pad,pad],[pad,pad],[0,0]])

def pad_oper(img, pad_size):
    # pad the image like pytorch with pad_size
    return tf.pad(img, padding_param(pad_size), "CONSTANT")

def conv_layer(img, weights, stride, padding, name, bias=False):
    with tf.variable_scope(name):
        # first we pad
        if padding!=0:
            pad_v = pad_oper(img, padding)
        else:
            pad_v = img
        # next we conv
        conv_v = tf.nn.conv2d(pad_v, weights[name+'.weight'], [1, stride, stride, 1], padding='VALID')
        # last we bias
        if bias != False:
            bias_v = tf.nn.bias_add(conv_v, weights[name+'.bias'])
            return bias_v
        else:
            return conv_v

def bn_layer(img, weights, name):
    return  tf.nn.batch_normalization(
                                        img,
                                        mean=weights[name+'.running_mean'],
                                        variance=weights[name+'.running_var'],
                                        offset=weights[name+'.bias'],
                                        scale=weights[name+'.weight'],
                                        variance_epsilon=1e-05,
                                        name=name
                                )


def max_pool(img, kernel_size, stride, padding, name):
    with tf.variable_scope(name):
        # first we pad
        pad_v = pad_oper(img, padding)
        # next we maxpool
        return tf.nn.max_pool(pad_v, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)


def downsample(img, weights,stride, name):
    '''
    mimic the downsample process
    '''
    with tf.variable_scope(name):
        conv = conv_layer(img, weights, stride=stride, padding=0, bias=False, name=name+'.0')
        bn = bn_layer(conv, weights, name = name+'.1')
        return bn



def bottleneck(img, weights, stride, name, downsample_flag=False):
    '''
    this function aims to mimic the Bottleneck class in pytorch
    example weight name: layer1.0.conv1.weight
    input name style example: layer1.0
    
    '''
    with tf.variable_scope(name):
        if downsample_flag is not False:
            residual = downsample(img, weights,stride=stride, name = name+'.downsample')
            
        else:
            residual = img
        conv1 = conv_layer(img, weights, stride=1, padding=0, bias=False, name=name+'.conv1')

        

        bn1 = bn_layer(conv1, weights, name=name+'.bn1')

        relu1 = tf.nn.relu(bn1)

        conv2 = conv_layer(relu1, weights, stride=stride, padding=1, bias=False, name=name+'.conv2')
        bn2 = bn_layer(conv2, weights, name=name+'.bn2')
        relu2 = tf.nn.relu(bn2)

        conv3 = conv_layer(relu2, weights, stride=1, padding=0, bias=False, name=name+'.conv3')
        bn3 = bn_layer(conv3, weights, name=name+'.bn3')

        out_sum = bn3 + residual
        out = tf.nn.relu(out_sum)
        return out
