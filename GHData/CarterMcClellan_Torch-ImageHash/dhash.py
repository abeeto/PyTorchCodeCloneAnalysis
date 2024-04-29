import torch
from torchvision import transforms
import torch.nn as nn

import PIL

class DHasher(nn.Module):
    # torch-scriptable variant
    def __init__(self, hash_size = 8):
        super().__init__()
        self.grayscale = transforms.Grayscale()
        self.resize = transforms.Resize(size=(hash_size, hash_size + 1),)
    
    def forward(self, X: torch.Tensor):
        # grayscale + crop image
        X = self.grayscale(X)
        X = self.resize(X).float()[0]
        
        return X[:, 1:] > X[:, :-1]

class DHasherPIL(nn.Module):
    # module should have exact parity w/ 
    #     https://github.com/JohannesBuchner/imagehash/blob/master/imagehash.py#L170
    def __init__(self, hash_size = 8):
        super().__init__()        
        self.img_size = hash_size
        self.trans = transforms.Compose([transforms.PILToTensor()])
    
    def forward(self, pil_img: PIL.Image):
        # apply transforms in PIL
        pil_img = pil_img.convert("L").resize((self.img_size + 1, self.img_size), PIL.Image.ANTIALIAS)
        
        # convert to tensor
        X = self.trans(pil_img).float()[0]
        
        return X[:, 1:] > X[:, :-1]
