import numpy as np
import torch
from .hooks import register_forward_hooks


def _get_module_info(mod):
    # format: kxk/s/p/d:<dilation>/g:<group>
    strs = []
    try:
        if hasattr(mod, 'kernel_size'):
            strs.append('x'.join(map(str, mod.kernel_size)))
        if hasattr(mod, 'stride'):
            strs.append(str(mod.stride[0]) if mod.stride[0] == mod.stride[1] else 'x'.join(map(str, mod.stride)))
        if hasattr(mod, 'padding'):
            strs.append(str(mod.padding[0]) if mod.padding[0] == mod.padding[1] else 'x'.join(map(str, mod.padding)))
        if hasattr(mod, 'dilation'):
            if max(mod.dilation) > 1:
                strs.append('d:'+str(mod.dilation[0]) if mod.dilation[0] == mod.dilation[1] else 'x'.join(map(str, mod.dilation)))
        if hasattr(mod, 'groups'):
            if mod.groups > 1:
                strs.append('g:'+str(mod.groups))
        if hasattr(mod, 'inplace') and mod.inplace:
            strs.append('inplace')
    except Exception as e:
        print(e)
    return '/'.join(strs)


def _module_type(mod, inputs, output):
    return ':'.join([type(mod).__name__, _get_module_info(mod)])

def _is_leaf(mod, inputs, output):
    return len(mod._modules) == 0

def _output_shape(mod, inputs, output):
    if torch.is_tensor(output):
        return tuple(output.shape)
    else:
        return () # not collected tensors in custom format

def _param_shape(mod, inputs, output):
    return [tuple(p.shape) for p in mod.parameters(recurse=False)]

def _param_num(mod, inputs, output):
    # if mod._modules:
    #     return 0  # not collected for non-leaf modules # NOTE: custom non-leaf module can have parameters
    return sum([param.numel() for param in mod.parameters(recurse=False)])

def _flops_basic(mod, inputs, output):
    """Calcluate FLOPs (multiply-adds) from module, inputs, and output.

    NOTE: only calculate _ConvNd and Linear layers."""
    shape = _output_shape(mod, inputs, output)
    if isinstance(mod, torch.nn.modules.conv._ConvNd):
        # (N*C_{l+1}*H*W)*C_l*K*K/G
        res = 1.0 * np.prod(shape) * mod.in_channels * np.prod(mod.kernel_size) / mod.groups
    elif isinstance(mod, torch.nn.Linear):
        res = 1.0 * np.prod(shape) * mod.in_features
    else:
        res = 0
    return int(res)

def __flops_rnn_one_layer(D, length, batch_size, H_in, H_out, bias):
    """Refer to nn.RNN.__doc__ for implementation details."""
    L, N = length, batch_size
    b = 0.5 if bias else 0.0
    # N*H_out*H_in for w_ih, N*H_out*H_out for w_hh, 2*N*H_out for tanh
    return D*L*(N*H_out*(H_in+b) + N*H_out*(H_out+b) + 2*N*H_out)

def __flops_rnn(num_layers, D, length, batch_size, H_in, H_hidden, H_out, bias):
    H1s = [H_in] + [H_hidden] * (num_layers-1)
    H2s = [H_hidden] * (num_layers-1) + [H_out]
    res = 0.0
    # D = 2 if bidiretional else 1
    for H1, H2 in zip(H1s, H2s):
        res += __flops_rnn_one_layer(D, length, batch_size, H1, H2, bias)
    return res    

def _flops_full(mod, inputs, output):
    """Calcluate FLOPs (multiply-adds) from module, inputs, and output. (Alternative method)

    NOTE: Original method does not count bias and normalization (typically match the reports in publications), 
    Alternative method is more accurate (we treat multiply, add, divide, exp the same for calculation)."""
    shape = _output_shape(mod, inputs, output)
    if isinstance(mod, torch.nn.modules.conv._ConvNd):
        # (N*C_{l+1}*H*W)*C_l*K*K/G
        res = 1.0 * np.prod(shape) * mod.in_channels * np.prod(mod.kernel_size) / mod.groups
        if mod.bias is not None:
            res += np.prod(shape)
    elif isinstance(mod, torch.nn.Linear):
        res = 1.0 * np.prod(shape) * mod.in_features
        if mod.bias is not None:
            res += np.prod(shape)
    elif isinstance(mod, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)):
        # out = (in-mean[c])/sqrt(var[c])*gamma[c]+beta[c]
        # Note: BatchNormXd, GroupNorm use affine, LayerNorm use elmentwise_affine
        if mod.weight is not None and mod.bias is not None:
            res = 2.0 * np.prod(shape) # treat divide as multiply, so two multiply-adds (mean-var and gamma-beta)
        else:
            res = 1.0 * np.prod(shape)
    elif isinstance(mod, torch.nn.RNNBase): # for RNN, LSTM, GRU
        num_layers = mod.num_layers
        D = 2 if mod.bidirectional else 1
        H_in = mod.input_size
        H_hidden = H_out = mod.hidden_size
        bias = mod.bias
        input = inputs[0]
        batch_size = input.shape[0] if mod.batch_first else input.shape[1]
        length = input.shape[1] if mod.batch_first else input.shape[0]
        
        rnn_flops = __flops_rnn(num_layers, D, length, batch_size, H_in, H_hidden, H_out, bias)
        
        if isinstance(mod, torch.nn.RNN):
            res = rnn_flops
        elif isinstance(mod, torch.nn.LSTM):
            res = 4 * rnn_flops # LSTM is 4x
            # 3: 3 multiply + 1 add, 2: tanh
            res += (num_layers * D * length * (3+2) * (batch_size * H_hidden)) # for sigmoid, tanh, etc.
        elif isinstance(mod, torch.nn.GRU):
            res = 3 * rnn_flops # GRU is 3x
            # 3: 3 multiply, ignore add and substract
            res += (num_layers * D * length * 3 * (batch_size * H_hidden)) # for sigmoid, tanh, etc.
        else:
            res = 0 # invalid RNN
    elif type(mod).__name__ == 'Attention':
        try:
            # import timm
            # assert isinstance(mod, timm.models.vision_transformer.Attention)
            # Only for timm.models.vision_transformer.Attention
            # Attention has three parts: qkv, attn_aggr, proj. The first and last part is calculated in nn.Linear, we need to add attn_aggr FLOPs (which is done as functional, and is not tracked by module)
            B, N, Cout = shape; Cin = mod.qkv.weight.shape[0]
            H = mod.num_heads; Cval = Cin // H
            # N*Cval*N for sim=q.dot(k), 2*N*N for softmax, N*N*Cval for sim.dot(v)
            # for softmax, exp with divide treat as one multipy-add
            res = 1.0 * B * H * (N * Cval * N + 2 * N * N + N * N * Cval) 
        except Exception as e:
            res = 0
    else:
        res = 0
    return int(res)

# def _mem(mod, inputs, output):
#     import numpy as np
#     if hasattr(mod, 'inplace') and mod.inplace == True:
#         res = 0
#     else:
#         shape = _output_shape(mod, inputs, output)
#         res = 1.0 * np.prod(shape) * 4  # (in bytes, 1 float is 4 bytes)
#     return int(res)

@torch.no_grad()
def print_summary(model, *inputs, **kwargs):
    # use as torchtools.utils.print_summary
    NOTE = """NOTE:
    *: leaf modules
    Flops is measured in multiply-adds. Multiply, divide, exp are treated the same for calculation, add is ignored except for bias.
    Flops (basic) only calculates for convolution and linear layers (not inlcude bias)
    Flops additionally calculates for bias, normalization (BatchNorm, LayerNorm, GroupNorm), RNN (RNN, LSTM, GRU) and attention layers
        - activations (e.g. ReLU), operations implemented as functionals (e.g. add in a residual architecture) are not 
          calculated as they are usually neglectable.
        - complex custom module may need manual calculation for correctness (refer to RNN, LSTM, GRU, Attention as examples).
    """
    with register_forward_hooks(model) as forward:
        model.eval()
        outputs = model(*inputs, **kwargs)
        forward.register_extra_hook('module_type', _module_type)
        forward.register_extra_hook('is_leaf', _is_leaf)
        forward.register_extra_hook('output_shape', _output_shape)
        forward.register_extra_hook('param_shape', _param_shape)
        forward.register_extra_hook('param_num', _param_num)
        forward.register_extra_hook('flops_basic', _flops_basic)
        forward.register_extra_hook('flops_full', _flops_full)
        # forward.register_extra_hook('mem', _mem)

        def print_line(col_names, col_limits):
            print(' '.join([('{:>%d}' % n).format(s) for n, s in zip(col_limits, col_names)]))

        col_names = ['Layer (type)', 'Output shape', 'Param shape', 'Param #', 'FLOPs basic', 'FLOPs'] #, 'Memory (B)']
        col_limits = [40, 15, 15, 12, 15, 15] #, 12]
        total_limit = sum(col_limits) + len(col_limits) - 1
        # print summary head
        print(('-' * total_limit))
        print_line(col_names, col_limits)
        print(('=' * total_limit))
        # print inputs
        for x in inputs:
            if hasattr(x, 'shape'): # torch.Tensor or ndarray
                col_names = ['Input' + ' *', 'x'.join(map(str, x.shape))]; print_line(col_names, col_limits)
        for name, x in kwargs.items():
            if hasattr(x, 'shape'): # torch.Tensor or ndarray
                col_names = ['Input ({})'.format(name) + ' *', 'x'.join(map(str, x.shape))]; print_line(col_names, col_limits)
        # print model leaf modules
        for _info in forward:
            col_names = ['{} ({})'.format(_info['module_name'], _info['module_type']) + (' *' if _info['is_leaf'] else '  '),
                'x'.join(map(str, _info['output_shape'])),
                '+'.join(['x'.join(map(str, shape)) for shape in _info['param_shape']]),
                '{:,}'.format(_info['param_num']),
                '{:,}'.format(_info['flops_basic']),
                '{:,}'.format(_info['flops_full']),
                # '{:,}'.format(_info['mem']),
                ]
            print_line(col_names, col_limits)
        print(('-' * total_limit))

        # print total
        total_params = sum([_info['param_num'] for _info in forward])
        total_params_with_aux = sum([p.numel() for p in model.parameters()])
        total_params_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
        total_params_non_trainable = sum([p.numel() for p in model.parameters() if not p.requires_grad])
        total_flops_basic = sum([_info['flops_basic'] for _info in forward])
        total_flops_full = sum([_info['flops_full'] for _info in forward])
        print(('Total params: {:,} ({:,} MB)'.format(total_params, total_params * 4 / (1024 * 1024))))
        print(('Total params (with aux): {:,} ({:,} MB)'.format(total_params_with_aux, total_params_with_aux * 4 / (1024 * 1024))))
        print(('    Trainable params: {:,} ({:,} MB)'.format(total_params_trainable, total_params_trainable * 4 / (1024 * 1024))))
        print(('    Non-trainable params: {:,} ({:,} MB)'.format(total_params_non_trainable, total_params_non_trainable * 4 / (1024 * 1024))))
        print(('Total flops (basic): {:,} ({:,} billion)'.format(total_flops_basic, total_flops_basic / 1e9)))
        print(('Total flops: {:,} ({:,} billion)'.format(total_flops_full, total_flops_full / 1e9)))
        print(('-' * total_limit))
        print(NOTE.rstrip())
        print(('-' * total_limit))

        res = {'flops': total_flops_full, 'flops_basic': total_flops_basic, 'params': total_params, 'params_with_aux': total_params_with_aux}
    return res

def test_print_summary():
    import torchvision.models as models
    model = models.resnet18()
    inputs = torch.randn(1,3,224,224)
    print_summary(model, inputs)
    
def test_print_summary_rnn():
    import torch.nn as nn
    
    rnn = nn.RNN(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
    print_summary(rnn, input, h0)
    
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print_summary(rnn, input, (h0, c0))
    
    rnn = nn.GRU(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
    print_summary(rnn, input, h0)
    
    # in the GRU example
    # H_in: 10, H_out: 20, num_layers: 2, length: 5, batch_size: 3
    # Param shape: 60x10+60x20+60+60+60x20+60x20+60+60
    #     w_ih_l0, w_hh_l0, bias, bias + w_ih_l1, w_hh_l1, bias, bias
    # major FLOPs: 5*(3*10*60+3*20*60) + 5*(3*20*60+3*20*60) = 63,000 
    # FLOPs: 70,200, the extra are from sigmoid or tanh
    
def test_print_summary_attn():
    import timm.models as models
    model = models.vit_base_patch16_224()
    inputs = torch.randn(1,3,224,224)
    print_summary(model, inputs)


if __name__ == "__main__":
    test_print_summary()
    test_print_summary_rnn()
    test_print_summary_attn()
