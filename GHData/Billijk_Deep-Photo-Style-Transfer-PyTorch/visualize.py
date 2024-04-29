import torch
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

masks = torch.load(sys.argv[1])
mask_tensors = [masks["tar"], masks["in"]]

for mask, save_root in zip(mask_tensors, ["style/", "content/"]):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    for tensor, idx in zip(mask, masks["categories"]):
        path = os.path.join(save_root, "{}.png".format(idx))
        plt.imsave(path, tensor.numpy())