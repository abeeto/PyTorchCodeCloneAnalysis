import numpy as np


# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class ImageBuffer:
    def __init__(self, max_buffer_size):
        self.buffer = []
        self.max_buffer_size = max_buffer_size

    def __call__(self, image):
        if len(self.buffer) < self.max_buffer_size:
            self.buffer.append(image)
            return image
        else:
            self.buffer.append(image)
            return_ind = np.random.randint(self.max_buffer_size)
            out = self.buffer[return_ind].clone()
            del self.buffer[return_ind]
            return out
