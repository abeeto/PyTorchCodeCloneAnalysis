import torch
from PIL import Image

def image_load_function(img_path):
        '''
        Given a path this will give you the image
        '''
        return Image.open(img_path)

def save(model,name,path):
        torch.save(model.state_dict(),path + name)