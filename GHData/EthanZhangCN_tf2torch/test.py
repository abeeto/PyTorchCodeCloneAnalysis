import torch
import vgg as VGG
import cv2
import numpy as np
import pdb
from numpy import linalg


def featcollapse(feat, new_size):
    feat_new = np.zeros((feat.shape[0], new_size))
    rate = int(feat.shape[0] / new_size)
    for i in range(feat_new.shape[1]):
        temp = feat[:, 2 * i].copy()
        for j in range(1, rate):
            temp = temp + feat[:, 2 * i + j]

        feat_new[:, i] = temp  # (feat[:,2*i] +feat[:,2*i+1])

    return feat_new


torch.set_grad_enabled(False)

dic_path = 'vgg16_from_tf_notop.pth'
torch_model = VGG.vgg16(pretrained=False)
state_dict =torch.load(dic_path)

# rm_keys = [ 'classifier.0.weight',
#             'classifier.0.bias',
#             'classifier.3.weight',
#             'classifier.3.bias',
#             'classifier.6.weight',
#             'classifier.6.bias',
# ]

# for i in range(6):
#     state_dict.pop(rm_keys[i])


torch_model.load_state_dict(state_dict,strict = True)

img = cv2.imread('3.jpg')
tc_img = np.expand_dims(np.transpose(img,(2,0,1)),0)
img_tensor = torch.Tensor(tc_img).cpu()
# print(img_tensor)
img_feat = torch_model(img_tensor).detach().numpy()
# print(img_feat)
# pdb.set_trace()
img_feat = featcollapse(img_feat,128)

norm_feat = np.zeros(img_feat.shape)
for i in range(img_feat.shape[0]):
    norm_feat[i, :] = img_feat[i, :] / linalg.norm(img_feat, axis=1)[i]

print(100*norm_feat.astype('float16'))

# ##tensorflow
# import cv2
# import numpy as np
# from keras.applications import VGG16
# from keras.models import Model

# from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# img = cv2.imread('3.jpg')
# tf_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)
# tf_model.predict(np.zeros((1, 224, 224, 3)))

# tf_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

# tf_img = np.expand_dims(img, axis=0)

# # tf_img = preprocess_input_vgg(tf_img)

# tf_feat = tf_model.predict(tf_img)

# print(tf_feat)