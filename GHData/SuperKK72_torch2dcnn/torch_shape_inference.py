# coding:utf-8
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch_classes import LPRNet
import json

def get_output_size(summary_dict, output):
    if isinstance(output, tuple):
        for i in range(len(output)):
            summary_dict[i] = OrderedDict()
            summary_dict[i] = get_output_size(summary_dict[i], output[i])
    else:
        summary_dict['output_shape'] = list(output.size())
    return summary_dict


def summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key] = get_output_size(summary[m_key], output)

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            # if hasattr(module, 'bias'):
            #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))

            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    return summary

if __name__ == '__main__':
    net = LPRNet.LPRNet(8, False, 68, 0)
    net_path = "./torch_models/LPRNet/weights/Final_LPRNet_model.pth"
    net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
    netStructure = summary([3, 24, 94], net)
    netStructureSimple = netStructure
    for info in netStructureSimple.values():
        if 'trainable' in info:
            info.pop('trainable')
        if 'nb_params' in info:
            info.pop('nb_params')
    for layer_name, info in netStructureSimple.items():
        print("layer name: {}\t".format(layer_name))
        for k, v in info.items():
            print('\t' + k, v)
    jsonOutput = json.dumps(netStructureSimple)
    # print(type(jsonOutput))
    jsonFilePath = "./torch_json/LPRNet.json"
    jsonFile = open(jsonFilePath, 'w', encoding='utf-8')
    json.dump(jsonOutput, jsonFile)




