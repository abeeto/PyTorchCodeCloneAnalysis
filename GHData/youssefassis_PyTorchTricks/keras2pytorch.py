# Loading weights of a pretrained keras model to PyTorch model

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization

import torch
from models import UNet3D

# Load keras weights 
kerasModel = load_model("path/to/keras/model_file.h5")

# Instance of PyTorch model 
pytorchmodel = UNet3D()

# Function to load weights of 'Conv3D, Conv3DTranspose and BatchNormalization' layers from keras to pytorch
def keras_to_pytorch(keras, pytorch=None):
    weight_dict = dict()
    for layer in keras.layers:
        if (type(layer) is Conv3D) or (type(layer) is Conv3DTranspose):
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (4,3,0,1,2))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif (type(layer) is BatchNormalization):
            weight_dict[layer.get_config()['name'] + '.weight'] = layer.get_weights()[0]
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            weight_dict[layer.get_config()['name'] + '.running_mean'] = layer.get_weights()[2]
            weight_dict[layer.get_config()['name'] + '.running_var'] = layer.get_weights()[3]
    if pytorch is not None:
        state_dict = pytorch.state_dict()
        st = [key for key in state_dict if "num_batches_tracked" not in  key]
        for p, k in zip(st, weight_dict):
            keras_weight = torch.from_numpy(weight_dict[k])
            assert state_dict[p].shape == keras_weight.shape, f"Error in shapes when assigning weights to '{p}'" 
            state_dict[p] = keras_weight
        pytorch.load_state_dict(state_dict)
        print("Weights are loaded successfully (from Keras to PyTorch)")
        return pytorch
    return weight_dict
  
# Test
keras_to_pytorch(kerasModel, pytorchmodel)
