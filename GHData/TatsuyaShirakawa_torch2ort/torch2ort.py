import numpy as np
import torch


def _tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, )


def get_device(model):
    return next(iter(model.state_dict().values())).device


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif not isinstance(x, torch.Tensor):
        raise ValueError('np.ndarray or torch.Tensor are supported for inputs/outputs but got unsupported type {}'.format(type(x)))
    else:
        return x.to(device)


def log(*x, verbose=False):
    if verbose:
        print(*x)


def export(torch_model, sample_inputs,
           result_onnx_file,
           input_names=['input'],
           output_names=['output'],
           dynamic_axes=None,
           do_constant_folding=True,
           verbose=True,
           **export_args):
    assert(len(_tuple(sample_inputs)) == len(_tuple(input_names)))
    if dynamic_axes is not None:
        assert(type(dynamic_axes) == dict)
        assert(all([type(k) == str and type(v) == list for k, v in dynamic_axes.items()]))

    sample_inputs = _tuple(sample_inputs)
    input_names = _tuple(input_names)
    output_names = _tuple(output_names)
    
    log('shapes of sample_inputs:', [_.shape for _ in sample_inputs], verbose=verbose)
    log('result_onnx_file:', result_onnx_file, verbose=verbose)
    log('input_names:', input_names, verbose=verbose)
    log('output_names:', output_names, verbose=verbose)
    log('dynamic_axes:', dynamic_axes, verbose=verbose)
    log('do_constant_folding:', do_constant_folding, verbose=verbose)
    for k, v in export_args.items():
        log('{}:'.format(k), v)

    device = get_device(torch_model)
    log('device: {}'.format(device), verbose=verbose)

    sample_inputs = tuple([to_tensor(_, device) for _ in sample_inputs])

    log('export ...', verbose=verbose)
    torch.onnx.export(torch_model,
                      sample_inputs,
                      result_onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      do_constant_folding=do_constant_folding,
                      **export_args)

    log('model is successfully exported to {}'.format(result_onnx_file), verbose=verbose)
