import torch.nn as nn


def modify_first_conv_in_channels(model, new_in_channels=1):
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            print('Found first conv layer (%s): %s' % (n, m))
            if m.in_channels != new_in_channels:
                # Get objects hierarchically
                hiers = n.split('.')
                levels = [model]
                for hier in hiers[:-1]:
                    levels.append(getattr(levels[-1], hier))
                # Assign new 
                conv_class = m.__class__(new_in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, False if m.bias==None else True, m.padding_mode)
                conv_class.load_state_dict(modify_weight(m.state_dict(), new_in_channels))
                setattr(levels[-1], hiers[-1], conv_class)
            break
    
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            print('Modified first conv layer (%s) to: %s' % (n, m))
            break


def modify_weight(state_dict, new_in_channels):
    new_state_dict = {}
    for key in state_dict:
        if key == 'weight':
            new_state_dict[key] = state_dict['weight'].sum(dim=1, keepdim=True).repeat(1, new_in_channels, 1, 1) / new_in_channels
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def test(model_str, new_in_channels):
    import torch
    import torchvision
    model = torchvision.models.__dict__[model_str](pretrained=True)
    model.eval()
    dummy_data = torch.randn(1, 1, 32, 32)
    for n,m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            sd = m.state_dict()
            ch_duplicated_data = dummy_data.repeat(1,m.in_channels,1,1)
            with torch.no_grad():
                orig_conv_out = m(ch_duplicated_data)
                orig_model_out = model(ch_duplicated_data)
            break
    
    modify_first_conv_in_channels(model, new_in_channels)
    
    for n,m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            new_sd = m.state_dict()
            ch_duplicated_data = dummy_data.repeat(1,m.in_channels,1,1)
            with torch.no_grad():
                new_conv_out = m(ch_duplicated_data)
                new_model_out = model(ch_duplicated_data)
            assert(new_sd['weight'].shape[1] == new_in_channels)
            assert(torch.equal(new_sd['weight'][:, :1, :, :], sd['weight'].sum(dim=1, keepdim=True)/new_in_channels))
            print('1st Conv Max Error:', (orig_conv_out - new_conv_out).abs().max())
            print('1st Conv Mean Absolute Error:', (orig_conv_out - new_conv_out).abs().mean())
            #assert(torch.allclose(orig_conv_out, new_conv_out, atol=1e-05))
            print('Model Max Error:', (orig_model_out - new_model_out).abs().max())
            print('Model Mean Absolute Error:', (orig_model_out - new_model_out).abs().mean())
            #assert(torch.allclose(orig_model_out, new_model_out, atol=1e-05))
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: %s [model] [in_channels]' % (sys.argv[0]))
    test(sys.argv[1], int(sys.argv[2]))
