import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import data_loader


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    d = dir

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class RGBDFolder(data.Dataset):

    def __init__(
            self,
            root,
            root_depth,
            transform_rgb=None,
            transform_depth=None,
            transform_mutual=None,
            loader_rgb=data_loader.pil_loader,
            loader_depth=data_loader.torch_tensor_loader
    ):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )

        if not root_depth.endswith('/'):
            root_depth += '/'

        self.root = root
        self.root_depth = root_depth
        self.imgs = imgs

        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.transform_mutual = transform_mutual

        self.loader_rgb = loader_rgb
        self.loader_depth = loader_depth
        return

    def __getitem__(self, index=0):
        # load rgb images
        path_rgb = self.imgs[index]
        img = self.loader_rgb(path_rgb)

        # perform transforms on original rgb image
        if self.transform_rgb is not None:
            if self.transform_mutual is not None:
                img, img_disp = self.transform_rgb(img, self.transform_mutual)
            else:
                img_disp = torch.from_numpy(np.transpose(img, [2, 0, 1]))
                img = self.transform_rgb(img)

        # load depth map and perform transforms on it
        path_depth = self.root_depth
        path_depth += str(int(path_rgb[len(path_rgb)-9:len(path_rgb)-4]))
        path_depth += '.pth'
        depth = self.loader_depth(path_depth)
        if self.transform_depth is not None:
            if self.transform_mutual is not None:
                depth = self.transform_depth(depth, self.transform_mutual)
            else:
                depth = self.transform_depth(depth)

        return img, depth, img_disp

    def __len__(self):
        return len(self.imgs)
