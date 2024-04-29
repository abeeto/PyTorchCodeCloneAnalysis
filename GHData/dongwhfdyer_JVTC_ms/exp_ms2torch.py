from mindspore import save_checkpoint, Parameter, Tensor
from mindspore.common.initializer import initializer


def torch_to_ms(model, torch_model, save_path):
    """
    Updates mobilenetv2 model mindspore param's data from torch param's data.
    Args:
        model: mindspore model
        torch_model: torch model
    """
    print("start load")
    # load torch parameter and mindspore parameter
    torch_param_dict = torch_model
    ms_param_dict = model.parameters_dict()
    count = 0
    for ms_key in ms_param_dict.keys():
        ms_key_tmp = ms_key.split('.')
        if ms_key_tmp[0] == 'bert_embedding_lookup':
            count += 1
            update_torch_to_ms(torch_param_dict, ms_param_dict, 'embeddings.word_embeddings.weight', ms_key)

        elif ms_key_tmp[0] == 'bert_embedding_postprocessor':
            if ms_key_tmp[1] == "token_type_embedding":
                count += 1
                update_torch_to_ms(torch_param_dict, ms_param_dict, 'embeddings.token_type_embeddings.weight', ms_key)
            elif ms_key_tmp[1] == "full_position_embedding":
                count += 1
                update_torch_to_ms(torch_param_dict, ms_param_dict, 'embeddings.position_embeddings.weight',
                                   ms_key)
            elif ms_key_tmp[1] == "layernorm":
                if ms_key_tmp[2] == "gamma":
                    count += 1
                    update_torch_to_ms(torch_param_dict, ms_param_dict, 'embeddings.LayerNorm.weight',
                                       ms_key)
                else:
                    count += 1
                    update_torch_to_ms(torch_param_dict, ms_param_dict, 'embeddings.LayerNorm.bias',
                                       ms_key)
        elif ms_key_tmp[0] == "bert_encoder":
            if ms_key_tmp[3] == 'attention':
                par = ms_key_tmp[4].split('_')[0]
                count += 1
                update_torch_to_ms(torch_param_dict, ms_param_dict, 'encoder.layer.' + ms_key_tmp[2] + '.' + ms_key_tmp[3] + '.'
                                   + 'self.' + par + '.' + ms_key_tmp[5],
                                   ms_key)
            elif ms_key_tmp[3] == 'attention_output':
                if ms_key_tmp[4] == 'dense':
                    print(7)
                    count += 1
                    update_torch_to_ms(torch_param_dict, ms_param_dict,
                                       'encoder.layer.' + ms_key_tmp[2] + '.attention.output.' + ms_key_tmp[4] + '.' + ms_key_tmp[5],
                                       ms_key)

                elif ms_key_tmp[4] == 'layernorm':
                    if ms_key_tmp[5] == 'gamma':
                        print(8)
                        count += 1
                        update_torch_to_ms(torch_param_dict, ms_param_dict,
                                           'encoder.layer.' + ms_key_tmp[2] + '.attention.output.LayerNorm.weight',
                                           ms_key)
                    else:
                        count += 1
                        update_torch_to_ms(torch_param_dict, ms_param_dict,
                                           'encoder.layer.' + ms_key_tmp[2] + '.attention.output.LayerNorm.bias',
                                           ms_key)
            elif ms_key_tmp[3] == 'intermediate':
                count += 1
                update_torch_to_ms(torch_param_dict, ms_param_dict,
                                   'encoder.layer.' + ms_key_tmp[2] + '.intermediate.dense.' + ms_key_tmp[4],
                                   ms_key)
            elif ms_key_tmp[3] == 'output':
                if ms_key_tmp[4] == 'dense':
                    count += 1
                    update_torch_to_ms(torch_param_dict, ms_param_dict,
                                       'encoder.layer.' + ms_key_tmp[2] + '.output.dense.' + ms_key_tmp[5],
                                       ms_key)

                else:
                    if ms_key_tmp[5] == 'gamma':
                        count += 1
                        update_torch_to_ms(torch_param_dict, ms_param_dict,
                                           'encoder.layer.' + ms_key_tmp[2] + '.output.LayerNorm.weight',
                                           ms_key)

                    else:
                        count += 1
                        update_torch_to_ms(torch_param_dict, ms_param_dict,
                                           'encoder.layer.' + ms_key_tmp[2] + '.output.LayerNorm.bias',
                                           ms_key)

        if ms_key_tmp[0] == 'dense':
            if ms_key_tmp[1] == 'weight':
                count += 1
                update_torch_to_ms(torch_param_dict, ms_param_dict,
                                   'pooler.dense.weight',
                                   ms_key)
            else:
                count += 1
                update_torch_to_ms(torch_param_dict, ms_param_dict,
                                   'pooler.dense.bias',
                                   ms_key)
        else:
            count += 1
            update_torch_to_ms(torch_param_dict, ms_param_dict,
                               ms_key,
                               ms_key)

    save_checkpoint(model, save_path)
    print("finish load")


def update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp):
    """Updates mindspore batchnorm param's data from torch batchnorm param's data."""
    str_join = '.'
    if ms_key_tmp[-1] == "moving_mean":
        ms_key_tmp[-1] = "running_mean"
        torch_key = str_join.join(ms_key_tmp)  # layer1.0.bn1.moving_mean
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
        update_torch_to_ms(torch_param_dict, ms_param_dict, ms_key, ms_key)


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


import BertConfig
import BertModel as ms_bm
import BertModel as tc_bm

bert_config_file = "./model/test.yaml"
bert_config = BertConfig.from_yaml_file(bert_config_file)
model = ms_bm(bert_config, False)

torch_model = tc_bm.from_pretrained("/content/model/bert_cn")
torch_to_ms(model, torch_model.state_dict(), "./model/bert2.ckpt")
