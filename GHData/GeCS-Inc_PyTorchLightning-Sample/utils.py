import numpy as np
from PIL import Image


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name: str):
        del self[name]


def plot_dataset(dataset, size=384, row=2, col=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    preview = np.zeros((row*size, col*size, 3), dtype='uint8')
    for i, (img, _) in enumerate(dataset):
        if i >= row * col:
            break
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * std + mean) * 255
        img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype("uint8")).resize((size, size))
        img = np.array(img)
        x, y = i % col, i // col
        preview[y*size:(y+1)*size, x*size:(x+1)*size] = img
    Image.fromarray(preview).save("dataset-sample.png")
