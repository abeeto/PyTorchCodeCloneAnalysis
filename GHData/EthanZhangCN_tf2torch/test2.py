import torch
import vgg as VGGs
import cv2
import numpy as np
import pdb


torch.set_grad_enabled(False)

torch_model = VGGs.vgg16(pretrained=True)

img = cv2.imread('./3.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = np.expand_dims(np.transpose(img,(2,0,1)),0)


img_tensor = torch.Tensor(img).cpu()

img_feat = torch_model(img_tensor).detach().numpy()


print(img_feat)
