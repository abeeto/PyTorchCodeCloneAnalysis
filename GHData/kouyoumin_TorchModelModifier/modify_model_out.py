import math
import torch.nn as nn


def modify_classifier_out(model, out_channels=1, selected_channels=None):
    for n, m in model.named_children():
        if n == 'fc' or n == 'classifier':
            print('Found classifier (%s): %s' % (n, m))
            if isinstance(m, nn.modules.conv._ConvNd):
                cls_class = m.__class__(m.in_channels, out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, False if m.bias==None else True, m.padding_mode)
                cls_class.load_state_dict(_modify_weight(m.state_dict(), out_channels, selected_channels))
                setattr(model, n, cls_class)
            elif isinstance(m, nn.Linear):
                cls_class = m.__class__(m.in_features, out_channels, False if m.bias==None else True)
                cls_class.load_state_dict(_modify_weight(m.state_dict(), out_channels, selected_channels))
                setattr(model, n, cls_class)
            elif isinstance(m, nn.Sequential):
                for idx in reversed(range(len(m))):
                    if isinstance(m[idx], nn.modules.conv._ConvNd):
                        cls_class = m[idx].__class__(m[idx].in_channels, out_channels, m[idx].kernel_size, m[idx].stride, m[idx].padding, m[idx].dilation, m[idx].groups, False if m[idx].bias==None else True, m[idx].padding_mode)
                        cls_class.load_state_dict(_modify_weight(m[idx].state_dict(), out_channels, selected_channels))
                        m[idx] = cls_class
                        break
                    elif isinstance(m[idx], nn.Linear):
                        cls_class = m[idx].__class__(m[idx].in_features, out_channels, False if m[idx].bias==None else True)
                        cls_class.load_state_dict(_modify_weight(m[idx].state_dict(), out_channels, selected_channels))
                        m[idx] = cls_class
                        break
                
            break
    
    for n, m in model.named_children():
        if n == 'fc' or n == 'classifier':
            print('Modified classifier (%s) to: %s' % (n, m))
            break


def _modify_weight(state_dict, out_channels, selected_channels=None):
    new_state_dict = {}
    for key in state_dict:
        original_out_channels = state_dict[key].shape[0]
        if isinstance(selected_channels, (tuple, list)) and len(selected_channels) == out_channels and out_channels <= original_out_channels:
            new_state_dict[key] = state_dict[key][selected_channels, ...]
        elif out_channels > original_out_channels:
            param_dims = state_dict[key].ndim
            repeats = [math.ceil(out_channels / original_out_channels)] + [1] * (param_dims - 1)
            new_state_dict[key] = state_dict[key].repeat(*repeats)[:out_channels, ...]
        else:
            new_state_dict[key] = state_dict[key][:out_channels, ...]
        '''if key.endswith('weight'):
            if isinstance(selected_classes, (tuple, list)) and len(selected_classes) == out_channels:
                new_state_dict[key] = state_dict[key][selected_classes, ...]
            else:
                new_state_dict[key] = state_dict[key][:out_channels, ...]
        else:
            if isinstance(selected_classes, (tuple, list)) and len(selected_classes) == out_channels:
                new_state_dict[key] = state_dict[key][]'''
            
    return new_state_dict


def test(model_str, new_out_features):
    import torch
    import torchvision
    model = torchvision.models.__dict__[model_str](pretrained=True)
    model.eval()
    #dummy_data = torch.randn(1, 1, 32, 32)
    for n,m in model.named_children():
        if n == 'fc' or n == 'classifier':
            sd = m.state_dict()
            break
    
    modify_classifier_out(model, new_out_features, selected_channels=None if new_out_features<1000 else list(range(new_out_features)))
    
    for n,m in model.named_modules():
        if n == 'fc' or n == 'classifier':
            new_param = list(m.parameters())
            assert(new_param[-1].shape[0] == new_out_features)
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: %s [model] [out_features]' % (sys.argv[0]))
    test(sys.argv[1], int(sys.argv[2]))
