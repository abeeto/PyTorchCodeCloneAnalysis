import torch
from torchvision import transforms
import torch.nn as nn

import PIL

class PHasher(nn.Module):
    # module is torchscriptable
    def __init__(self,):
        super().__init__()
        self.hash_size = 8
        highfreq_factor = 4
        img_size = self.hash_size * highfreq_factor
        self.grayscale = transforms.Grayscale()
        self.resize = transforms.Resize(size=(img_size, img_size),)
        self.pi = 3.1415927410125732
        
    def dct(self, x: torch.Tensor, axis: int):
        if axis != -1:
            x = torch.transpose(x, axis, -1)

        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * self.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        V = 2 * V.view(x_shape[0], x_shape[1])

        if axis != -1:
            V = torch.transpose(V, axis, -1)

        return V.float()
    
    def forward(self, X: torch.Tensor):
        # grayscale + crop image
        X = self.grayscale(X)
        X = self.resize(X).float()[0]
        
        # compute dct across rows + columns
        X = self.dct(X, axis=0)
        dct_result = self.dct(X, axis=1)
        
        # crop to upper left hand corner
        dctlowfreq = dct_result[:self.hash_size, :self.hash_size]

        # median binarize result
        med = torch.median(dctlowfreq)
        return dctlowfreq > med

class PHasherPIL(nn.Module):
    # module should have exact parity w/ 
    #     https://github.com/JohannesBuchner/imagehash/blob/master/imagehash.py#L197
    def __init__(self,):
        super().__init__()
        self.hash_size = 8
        highfreq_factor = 4
        self.img_size = self.hash_size * highfreq_factor
        self.trans = transforms.Compose([transforms.PILToTensor()])
        self.pi = 3.1415927410125732
        
    def dct(self, x: torch.Tensor, axis: int):
        if axis != -1:
            x = torch.transpose(x, axis, -1)

        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * self.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        V = 2 * V.view(x_shape[0], x_shape[1])

        if axis != -1:
            V = torch.transpose(V, axis, -1)

        return V.float()
    
    def forward(self, pil_img: PIL.Image):
        # apply transforms in PIL
        pil_img = pil_img.convert("L").resize((self.img_size, self.img_size), PIL.Image.ANTIALIAS)
        
        # convert to tensor
        X = self.trans(pil_img).float()[0]
        
        # compute dct across rows + columns
        dct_result = self.dct(self.dct(X, axis=0), axis=1)
        
        # crop to upper left hand corner
        dctlowfreq = dct_result[:self.hash_size, :self.hash_size]

        # median binarize result
        med = torch.median(dctlowfreq)
        return dctlowfreq > med
