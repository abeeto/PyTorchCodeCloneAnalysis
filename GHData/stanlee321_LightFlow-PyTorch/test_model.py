import unittest
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from model import LightFlow
import numpy as np

class Args:
    def __init__(self):
        self.rgb_max = 10
    
def r_imge(flying_chairs=True):
    if flying_chairs:
        image_a =  np.random.rand(384,512,3) * 255.
        image_b =  np.random.rand(384,512,3) * 255.
    else:
        image_a =  np.random.rand(256,256,3) * 255.
        image_b =  np.random.rand(256,256,3) * 255.
    
    return image_a, image_b

if __name__ == "__main__":

    args = Args()


    image_a, image_b = r_imge()

    image_a = torch.from_numpy(image_a)
    image_b = torch.from_numpy(image_b)
    
    image_a.unsqueeze_(0)
    image_b.unsqueeze_(0)

    image_a = image_a.permute(0, 3, 1, 2)
    image_b = image_b.permute(0, 3, 1, 2)

    print(image_a.shape)
    print(image_b.shape)
    
    with SummaryWriter(comment='LightFlow') as w:
        model = LightFlow(args)
        # Forward pass
        
        outputs = model(image_a.float(), image_b.float())
        output_size = (list(outputs.size())[2], list(outputs.size())[3])
        target = F.interpolate(image_a, size=output_size, mode='nearest')

        w.add_graph(model, (image_a.float(), image_b.float()), verbose=True)
        print('outputshape', target.shape)

    #print(model)