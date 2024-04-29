import numpy as np
import torch


def load_model(path):
    return torch.load(path)


class Worker:
    def __init__(self, path):
        self.path = path
        self.model = load_model(path)

    def process(self, img):
        # todo: ensure img is correct shape (1, 3, W, H)
        # todo: replace with proper preprocessing
        img = img.transpose((2, 0, 1))[None, ...].astype(np.float32) / 255.0
        with torch.no_grad():
            x = torch.tensor(img)
            out = self.model(x)
            out = out.data.numpy()
        out = (255.0 * out[0, ...]).clip(0, 255).astype(np.uint8)
        return out.transpose((1, 2, 0))
