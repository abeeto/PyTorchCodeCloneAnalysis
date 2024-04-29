import torch.nn as nn

convnd = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]
lazyconvnd = [None, nn.LazyConv1d, nn.LazyConv2d, nn.LazyConv3d]

convtransposend = [None, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
lazyconvtransposend = [None, nn.LazyConvTranspose1d, nn.LazyConvTranspose2d, nn.LazyConvTranspose3d]

maxpoolnd = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
avgpoolnd = [None, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]

adaptivemaxpoolnd = [None, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]
adaptiveavgpoolnd = [None, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]

batchnormnd = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
lazybatchnormnd = [None, nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d]

instancenormnd = [None, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
lazyinstancenormnd = [None, nn.LazyInstanceNorm1d, nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d]


def _modify_tuple_dim(original, target_size):
    if not isinstance(original, int):
        if len(original) >= target_size:
            return original[:target_size]
        else:
            return original[0]
    else:
        return original


def _modify_weight_dim(original, target_dim):
    new_sd = {}
    for key in original:
        if key.endswith('weight'):
            original_dim = original[key].ndim - 2
            num_diff_dim = target_dim - original_dim
            weight = original[key]
            if num_diff_dim > 0:
                for i in range(num_diff_dim):
                    repeats = [1] * weight.ndim + [weight.shape[-1]]
                    weight = weight.unsqueeze(-1).repeat(*repeats)
                    
                weight_correction_factor = weight.shape[-1] ** original_dim / weight.shape[-1] ** target_dim
                print(weight_correction_factor)
                weight = weight * weight_correction_factor
            elif num_diff_dim < 0:
                for i in range(abs(num_diff_dim)):
                    weight = weight.sum(dim=weight.ndim-1, keepdim=True).squeeze(-1)
            new_sd[key] = weight
            print('Changed', key, 'from', original[key].shape, 'to', new_sd[key].shape)
        else:
            new_sd[key] = original[key]
    return new_sd


def modify_model_dim(model, new_dim):
    assert(1 <= new_dim <= 3)
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvTransposeNd):
            print('Found conv layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)
            dilation = _modify_tuple_dim(m.dilation, new_dim)
            
            # Assign new
            if isinstance(m, nn.modules.conv._LazyConvXdMixin):
                conv_class = lazyconvtransposend[new_dim](m.out_channels, kernel_size, stride, padding, dilation, m.groups, False if m.bias==None else True, m.padding_mode)
                conv_class.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                setattr(levels[-1], hiers[-1], conv_class)
            else:
                conv_class = convtransposend[new_dim](m.in_channels, m.out_channels, kernel_size, stride, padding, dilation, m.groups, False if m.bias==None else True, m.padding_mode)
                conv_class.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                setattr(levels[-1], hiers[-1], conv_class)
        elif isinstance(m, nn.modules.conv._ConvNd):
            print('Found conv layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)
            dilation = _modify_tuple_dim(m.dilation, new_dim)

            # Assign new
            if isinstance(m, nn.modules.conv._LazyConvXdMixin):
                conv_class = lazyconvnd[new_dim](m.out_channels, kernel_size, stride, padding, dilation, m.groups, False if m.bias==None else True, m.padding_mode)
                conv_class.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                setattr(levels[-1], hiers[-1], conv_class)
            else:
                conv_class = convnd[new_dim](m.in_channels, m.out_channels, kernel_size, stride, padding, dilation, m.groups, False if m.bias==None else True, m.padding_mode)
                conv_class.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                setattr(levels[-1], hiers[-1], conv_class)
        elif isinstance(m, nn.modules.pooling._MaxPoolNd):
            print('Found pool layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)
            dilation = _modify_tuple_dim(m.dilation, new_dim)

            # Assign new 
            pool_class = maxpoolnd[new_dim]
            setattr(levels[-1], hiers[-1], pool_class(kernel_size, stride, padding, dilation, m.return_indices, m.ceil_mode))
        elif isinstance(m, nn.modules.pooling._AvgPoolNd):
            print('Found pool layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)

            # Assign new 
            pool_class = avgpoolnd[new_dim]
            setattr(levels[-1], hiers[-1], pool_class(kernel_size, stride, padding, m.ceil_mode, m.count_include_pad))
        elif isinstance(m, nn.modules.pooling._AdaptiveMaxPoolNd):
            print('Found pool layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            output_size = _modify_tuple_dim(m.output_size, new_dim)

            # Assign new 
            pool_class = adaptivemaxpoolnd[new_dim]
            setattr(levels[-1], hiers[-1], pool_class(output_size, m.return_indices))
        elif isinstance(m, nn.modules.pooling._AdaptiveAvgPoolNd):
            print('Found pool layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            output_size = _modify_tuple_dim(m.output_size, new_dim)

            # Assign new 
            pool_class = adaptiveavgpoolnd[new_dim]
            setattr(levels[-1], hiers[-1], pool_class(output_size))
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            print('Found batchnorm layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            # Assign new 
            if isinstance(m, nn.modules.batchnorm._LazyNormBase):
                norm_class = lazybatchnormnd[new_dim]
                setattr(levels[-1], hiers[-1], norm_class(m.eps, m.momentum, m.affine, m.track_running_stats))
            else:
                norm_class = batchnormnd[new_dim]
                setattr(levels[-1], hiers[-1], norm_class(m.num_features, m.eps, m.momentum, m.affine, m.track_running_stats))
        elif isinstance(m, nn.modules.instancenorm._InstanceNorm):
            print('Found instancenorm layer (%s): %s' % (n, m))
            hiers = n.split('.')
            levels = [model]
            for hier in hiers[:-1]:
                levels.append(getattr(levels[-1], hier))
            
            # Assign new 
            if isinstance(m, nn.modules.batchnorm._LazyNormBase):
                norm_class = lazyinstancenormnd[new_dim]
                setattr(levels[-1], hiers[-1], norm_class(m.eps, m.momentum, m.affine, m.track_running_stats))
            else:
                norm_class = instancenormnd[new_dim]
                setattr(levels[-1], hiers[-1], norm_class(m.num_features, m.eps, m.momentum, m.affine, m.track_running_stats))



def test(model_str, new_dim):
    import torch
    import torchvision
    model = torchvision.models.__dict__[model_str](pretrained=True)
    modify_model_dim(model, new_dim)
    print(model)

    dummy_data_size = [1,3]
    for i in range(new_dim):
        dummy_data_size.append(128)
    dummy_data = torch.randn(dummy_data_size)
    print('Testing model with dummy data:', dummy_data.shape)
    with torch.no_grad():
        out = model(dummy_data)
    print('Result tensor shape:', out.shape)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: %s [model] [new_dim]' % (sys.argv[0]))
    test(sys.argv[1], int(sys.argv[2]))