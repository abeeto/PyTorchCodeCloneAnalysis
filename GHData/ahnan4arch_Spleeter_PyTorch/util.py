import numpy as np
import tensorflow as tf

def tf2pytorch(checkpoint_path, num_instruments):
    tf_vars = {}
    init_vars = tf.train.list_variables(checkpoint_path)
    # print(init_vars)
    for name, shape in init_vars:
        try:
            # print('Loading TF Weight {} with shape {}'.format(name, shape))
            data = tf.train.load_variable(checkpoint_path, name)
            tf_vars[name] = data
        except Exception as e:
            print('Load error')
    conv_idx = 0
    tconv_idx = 0
    bn_idx = 0
    outputs = []
    for i in range(len(num_instruments)):
        output = {}
        outputs.append(output)

        for j in range(1,7):
            if conv_idx == 0:
                conv_suffix = ""
            else:
                conv_suffix = "_" + str(conv_idx)

            if bn_idx == 0:
                bn_suffix = ""
            else:
                bn_suffix = "_" + str(bn_idx)

            output['conv{}.weight'.format(j)] = np.transpose(
                tf_vars["conv2d{}/kernel".format(conv_suffix)], (3, 2, 0, 1))
            output['conv{}.bias'.format(
                j)] = tf_vars["conv2d{}/bias".format(conv_suffix)]

            output['encoder{}.0.weight'.format(
                j)] = tf_vars["batch_normalization{}/gamma".format(bn_suffix)]
            output['encoder{}.0.bias'.format(
                j)] = tf_vars["batch_normalization{}/beta".format(bn_suffix)]
            output['encoder{}.0.running_mean'.format(
                j)] = tf_vars['batch_normalization{}/moving_mean'.format(bn_suffix)]
            output['encoder{}.0.running_var'.format(
                j)] = tf_vars['batch_normalization{}/moving_variance'.format(bn_suffix)]

            conv_idx += 1
            bn_idx += 1

        # up blocks
        for j in range(1, 7):
            if tconv_idx == 0:
                tconv_suffix = ""
            else:
                tconv_suffix = "_" + str(tconv_idx)

            if bn_idx == 0:
                bn_suffix = ""
            else:
                bn_suffix= "_" + str(bn_idx)

            output['decoder{}.0.weight'.format(j)] = np.transpose(
                tf_vars["conv2d_transpose{}/kernel".format(tconv_suffix)], (3, 2, 0, 1))
            output['decoder{}.0.bias'.format(
                j)] = tf_vars["conv2d_transpose{}/bias".format(tconv_suffix)]
            output['decoder{}.2.weight'.format(
                j)] = tf_vars["batch_normalization{}/gamma".format(bn_suffix)]
            output['decoder{}.2.bias'.format(
                j)] = tf_vars["batch_normalization{}/beta".format(bn_suffix)]
            output['decoder{}.2.running_mean'.format(
                j)] = tf_vars['batch_normalization{}/moving_mean'.format(bn_suffix)]
            output['decoder{}.2.running_var'.format(
                j)] = tf_vars['batch_normalization{}/moving_variance'.format(bn_suffix)]
            tconv_idx += 1
            bn_idx += 1

        if conv_idx == 0:
            suffix = ""
        else:
            suffix = "_" + str(conv_idx)
        output['mask.weight'] = np.transpose(
            tf_vars['conv2d{}/kernel'.format(suffix)], (3, 2, 0, 1))
        output['mask.bias'] = tf_vars['conv2d{}/bias'.format(suffix)]
        conv_idx += 1
    
    return outputs
