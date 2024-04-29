from typing import Union

import numpy as np
from mindspore import save_checkpoint, Parameter, Tensor, load_param_into_net
from mindspore.common.initializer import initializer
import mindspore
# import torch


def generate_param_mapping_kuhn(m_net, tor_net, m_txt, t_txt, ms_ckpt):
    """
    save the parameter name and shape of mindspore and torch model in two txt file,
    and generate the ckpt file for mindspore model.
    """
    save_mindspore_net_txt(m_net, m_txt)
    save_torch_net_txt(tor_net, t_txt)
    par_dict = tor_net.state_dict()

    params_list = []
    f1 = open(m_txt, "r")
    f2 = open(t_txt, "r")

    lines_f1 = f1.readlines()
    lines_f2 = f2.readlines()
    assert lines_f1 != lines_f2, 'the two txt file is not equal,len(lines_f1)=%d,len(lines_f2)%d' % (len(lines_f1), len(lines_f2))
    for i in range(len(lines_f1)):
        param_dict = {}
        param_dict["name"] = lines_f1[i].strip()
        param_dict['data'] = Tensor(par_dict[lines_f2[i].strip()].numpy())
        params_list.append(param_dict)

    save_checkpoint(params_list, ms_ckpt)
    f1.close()
    f2.close()


def save_torch_net_txt(net, txt_path, include_shape=False, ):
    with open(txt_path, "w") as f:
        for key, value in net.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            if not include_shape:
                try:
                    f.write(str(key).strip() + "\n")
                except Exception as e:
                    print(e)
            else:
                f.write(str(key).strip() + " " + str(value.shape) + "\n")


def save_mindspore_net_txt(net, txt_path, include_shape=False):
    with open(txt_path, "w") as f:
        for item in net.get_parameters():
            if not include_shape:
                f.write(str(item.name).strip() + "\n")
            else:
                f.write(str(item.name).strip() + ' ' + str(item.shape) + '\n')


# def save_Tensor_as_txt(one_tensor: Union[torch.Tensor, mindspore.Tensor], txt_path):
#     """Save torch tensor to txt file."""
#     if isinstance(one_tensor, torch.Tensor):
#         np_array = one_tensor.numpy()
#     elif isinstance(one_tensor, mindspore.Tensor):
#         np_array = one_tensor.data.asnumpy()
#     else:
#         raise TypeError("one_tensor must be a torch.Tensor or mindspore.Tensor.")
#     with open(txt_path, "w") as f:
#         f.write(str(np_array))


def generate_param_mapping_ms(m_net, t_net, m_txt: str, t_txt: str, ckpt_ms: str):
    """Updates mindspore batchnorm param's data from torch batchnorm param's data."""
    ##########nhuk#################################### param save to txt
    torch_param_dict = t_net.state_dict()
    ms_param_dict = m_net.parameters_dict()
    save_mindspore_net_txt(m_net, m_txt, include_shape=True)
    save_torch_net_txt(t_net, t_txt, include_shape=True)
    ##########nhuk####################################

    for ms_key in ms_param_dict.keys():
        ms_key_tmp = ms_key.split('.')
        str_join = '.'

        if ms_key_tmp[-1] == "moving_mean":
            ms_key_tmp[-1] = "running_mean"
            torch_key = str_join.join(ms_key_tmp)
            update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
        elif ms_key_tmp[-1] == "moving_variance":
            ms_key_tmp[-1] = "running_var"
            torch_key = str_join.join(ms_key_tmp)
            update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
        elif ms_key_tmp[-1] == "gamma":
            ms_key_tmp[-1] = "weight"
            torch_key = str_join.join(ms_key_tmp)
            update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
        elif ms_key_tmp[-1] == "beta":
            ms_key_tmp[-1] = "bias"
            torch_key = str_join.join(ms_key_tmp)
            update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
        else:
            torch_key = str_join.join(ms_key_tmp)
            update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
        # load param from ms_param_dict to ms_net
    not_loading_list = load_param_into_net(net=m_net, parameter_dict=ms_param_dict)
    print("not_loading_list", not_loading_list)
    print("######################################## after loading param")
    print("checkpoint_path", ckpt_ms)

    save_checkpoint(m_net, ckpt_ms)


def update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key):
    """Updates mindspore param's data from torch param's data."""

    value = torch_param_dict[torch_key].cpu().numpy()
    value = Parameter(Tensor(value), name=ms_key)
    _update_param(ms_param_dict[ms_key], value)


def _update_param(param, new_param):
    """Updates param's data from new_param's data."""

    if isinstance(param.data, Tensor) and isinstance(new_param.data, Tensor):
        if param.data.dtype != new_param.data.dtype:
            print("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} type({}) different from parameter_dict's({})"
                   .format(param.name, param.data.dtype, new_param.data.dtype))
            raise RuntimeError(msg)

        if param.data.shape != new_param.data.shape:
            if not _special_process_par(param, new_param):
                print("Failed to combine the net and the parameters for param %s.", param.name)
                msg = ("Net parameters {} shape({}) different from parameter_dict's({})"
                       .format(param.name, param.data.shape, new_param.data.shape))
                raise RuntimeError(msg)
            return

        param.set_data(new_param.data)
        return

    if isinstance(param.data, Tensor) and not isinstance(new_param.data, Tensor):
        if param.data.shape != (1,) and param.data.shape != ():
            print("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} shape({}) is not (1,), inconsistent with parameter_dict's(scalar)."
                   .format(param.name, param.data.shape))
            raise RuntimeError(msg)
        param.set_data(initializer(new_param.data, param.data.shape, param.data.dtype))

    elif isinstance(new_param.data, Tensor) and not isinstance(param.data, Tensor):
        print("Failed to combine the net and the parameters for param %s.", param.name)
        msg = ("Net parameters {} type({}) different from parameter_dict's({})"
               .format(param.name, type(param.data), type(new_param.data)))
        raise RuntimeError(msg)

    else:
        param.set_data(type(param.data)(new_param.data))


def _special_process_par(par, new_par):
    """
    Processes the special condition.
    Like (12,2048,1,1)->(12,2048), this case is caused by GE 4 dimensions tensor.
    """
    par_shape_len = len(par.data.shape)
    new_par_shape_len = len(new_par.data.shape)
    delta_len = new_par_shape_len - par_shape_len
    delta_i = 0
    for delta_i in range(delta_len):
        if new_par.data.shape[par_shape_len + delta_i] != 1:
            break
    if delta_i == delta_len - 1:
        new_val = new_par.data.asnumpy()
        new_val = new_val.reshape(par.data.shape)
        par.set_data(Tensor(new_val, par.data.dtype))
        return True
    return False