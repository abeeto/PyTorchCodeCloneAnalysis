"""
Generate weight for two models
"""
import numpy as np
import pickle
import os


def get_torch_weights():
    if os.path.isfile('weights.pickle'):
        weights = pickle.load(open('weights.pickle', 'rb'))
    else:
        conv1_weight = np.random.normal(size=(10, 1, 5, 5))
        conv1_bias = np.random.normal(size=(10,))
        conv2_weight = np.random.normal(size=(20, 10, 5, 5))
        conv2_bias = np.random.normal(size=(20,))
        fc1_weight = np.random.normal(size=(50, 320))
        fc1_bias = np.random.normal(size=(50,))
        fc2_weight = np.random.normal(size=(10, 50))
        fc2_bias = np.random.normal(size=(10,))
        weights = (conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                   fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        pickle.dump(weights, open('weights.pickle', 'wb'))
    return weights


def get_tf_weights():
    (conv1_weight, conv1_bias, conv2_weight, conv2_bias,
     fc1_weight, fc1_bias, fc2_weight, fc2_bias) = get_torch_weights()
    tf_conv1_weight = np.transpose(conv1_weight, [2, 3, 1, 0])
    tf_conv2_weight = np.transpose(conv2_weight, [2, 3, 1, 0])
    tf_fc1_weight = np.transpose(fc1_weight)
    tf_fc2_weight = np.transpose(fc2_weight)
    return (tf_conv1_weight, conv1_bias, tf_conv2_weight, conv2_bias,
            tf_fc1_weight, fc1_bias, tf_fc2_weight, fc2_bias)
