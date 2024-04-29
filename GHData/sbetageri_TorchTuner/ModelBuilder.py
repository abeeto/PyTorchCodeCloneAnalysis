import torch
import torch.nn as nn

from collections import OrderedDict

def buildSequentialNetwork(layers=[], 
                         act_func=nn.ReLU,
                         output_act_func=nn.Softmax):
    '''Build a sequential neural network
    
    :param layers: List of dimensions of the layers in a neural network, defaults to []
    :param layers: list, optional
    :param act_func: activation function, defaults to nn.ReLU
    :param act_func: function, optional
    :param output_act_func: output activation function, defaults to nn.Softmax
    :param output_act_func: function, optional
    :return: Sequential Neural Network
    :rtype: nn.Sequential
    '''

    ## TODO
    ## Handle dropout layers. 

    seq_layers = OrderedDict()

    # Base names for layers
    fc_base_name = 'fc_'
    act_base_name = 'act_'
    output_name = 'output'

    for i in range(0, len(layers) - 1):
        fc_name = fc_base_name + str(i + 1)
        act_name = act_base_name + str(i + 1)
        seq_layers[fc_name] = nn.Linear(in_features=layers[i],
                                        out_features=layers[i + 1])
        if i != len(layers) - 2:
            seq_layers[act_name] = act_func()
        else:
            seq_layers[output_name] = output_act_func()
    
    seq_network = nn.Sequential(seq_layers)
    return seq_network


if __name__ == '__main__':
    hidden_layers = [512, 256, 128, 64, 32, 1000, 3]
    net = buildSequentialNetwork(layers=hidden_layers)
    print(net)