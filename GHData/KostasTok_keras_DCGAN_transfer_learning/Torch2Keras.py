# Initialise new keras dcgan model
import os
import numpy as np
import torch
from torch_dcgan import _netG
from keras_dcgan import DCGAN

def converter(torch_generator_path):
    '''
    torch_generator_path -> relative to this file, or absolute path
    
    1) Loads FloyedHub Torch model
    2) Creates new keras model and passes the torch weights to it
    3) Save weights to sub-directory weights (relative to this file)
    '''
    found = False
    if os.path.exists(torch_generator_path):
        weights_path = torch_generator_path
        found = True
    else:
        weights_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                torch_generator_path)
        if os.path.exists(weights_path):
            found = True
    
    if found:
        # Load pre-trained torch generator
        torch_model = _netG(True)

        torch_model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        print('The Torch Model:')
        print(torch_model.main)
        print('----------------------')

        # Pass weigths to numpy arrays and roll axis to match those of
        # the keras model
        weights = []
        for layer in torch_model.main:
            if 'weight' in dir(layer):
                if len(layer.weight.shape)==4:
                    w = layer.weight.detach().numpy()
                    w = np.moveaxis(w,2,0)
                    w = np.moveaxis(w,3,1)
                    w = np.moveaxis(w,3,2)
                    weights.append(w)

        # Load corresponding Keras model (untrained)
        print('The Keras model:')
        dcgan = DCGAN(generator='FloyedHub')
        print('----------------------')

        # Pass the weights from Torch to Keras model
        print('Passing weights to convolutional layers:')
        k = 0
        for layer in dcgan.generator.layers:
            try:
                layer.set_weights([weights[k]])
                print('weights set on: ', layer.name)
                k +=1
            except:
                print('no convolution: ', layer.name)
        
        # Saving the Keras model
        dcgan.save()
    else:
        print('Failed to find the file.')
