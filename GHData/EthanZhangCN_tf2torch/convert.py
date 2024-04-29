import torch
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model
import numpy as np

import pdb

# pytorch model
torch_dic = "vgg16-397923af.pth"
state_dict =torch.load(torch_dic)
print(state_dict.keys())


#tf model
tf_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)


# tf convert to torch
tflayers_name = [   'block1_conv1',
                    'block1_conv2',
                    'block2_conv1',
                    'block2_conv2',
                    'block3_conv1',
                    'block3_conv2',
                    'block3_conv3',
                    'block4_conv1',
                    'block4_conv2',
                    'block4_conv3',
                    'block5_conv1',
                    'block5_conv2',
                    'block5_conv3',
                ]

rm_keys = [ 'classifier.0.weight',
            'classifier.0.bias',
            'classifier.3.weight',
            'classifier.3.bias',
            'classifier.6.weight',
            'classifier.6.bias',
]


for i in range(6):
    state_dict.pop(rm_keys[i])

for i,key in enumerate(state_dict.keys()):
    k = int(i/2)

    tf_weights = tf_model.layers[[l.name==tflayers_name[k] for l in tf_model.layers].index(True)].get_weights()

    if (i % 2):
        tf_weights =tf_weights [1]
    else:

        tf_weights = tf_weights[0]
        tf_weights = np.swapaxes(tf_weights,0,-1)
        tf_weights = np.swapaxes(tf_weights,1,-2)

    print(k,":",key)
    state_dict[key] = torch.Tensor(tf_weights)

torch.save(state_dict,'vgg16_from_tf_notop.pth')

pdb.set_trace()

