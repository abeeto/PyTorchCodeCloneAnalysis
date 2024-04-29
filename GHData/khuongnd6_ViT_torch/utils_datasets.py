# %%
import enum
from numpy.lib.arraysetops import isin
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
from torch._C import Value
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import time
# from utils_datasets import Cutout, CIFAR10Policy, ImageNetPolicy

# %%
from skimage.feature import local_binary_pattern
import cv2
from torchvision.transforms.transforms import Normalize

# %%
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class STL10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on ImageNet.
        Example:
        >>> policy = STL10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     STL10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment STL10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: self._rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    @classmethod
    def _rotate_with_fill(cls, img, magnitude):
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


# %%
class CustomRepresentations(torch.utils.data.Dataset):
    """Custom frozen image representations dataset."""

    def __init__(self,
                reprs=None,
                labels=None,
                transform=None,
                **kwargs):
        """
        Args:
        """
        self.transform = transform
        self.reprs = reprs
        self.labels = labels
        
    def __len__(self):
        return len(self.reprs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        _repr = self.reprs[idx]
        _label = self.labels[idx]
        if self.transform:
            sample = [self.transform(transforms.ToPILImage(_repr)), _label]
        else:
            sample = [_repr, _label]
        
        
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        return sample

class CustomPostProcessDatasets:
    def __init__(self,
                train_reprs=None,
                train_labels=None,
                test_reprs=None,
                test_labels=None,
                splits=['train', 'test'],
                limit_train=0,
                limit_test=0,
                num_workers=4,
                shuffle=False,
                bs=128,
                transform=[],
                norm_values=None,
                num_labels=5,
                ):
        self.num_labels = num_labels
        self.info = {
            'batch_count': {},
            'sample_count': {},
        }
        self.sets = {}
        self.loaders = {}
        norm_transform = []
        if norm_values:
            norm_transform = [transforms.Normalize(**norm_values)]
        for _split in splits:
            data_kwargs = {
                'train': {
                    'reprs': train_reprs,
                    'labels': train_labels,
                },
                'test': {
                    'reprs': test_reprs,
                    'labels': test_labels,
                },
            }
            _set = CustomRepresentations(
                **data_kwargs.get(_split, {}),
                transform=transforms.Compose([
                    *transform,
                    transforms.ToTensor(),
                    *norm_transform,
                ])
            )
            _limit = {
                'train': limit_train,
                'test': limit_test,
            }[_split]
            if isinstance(_limit, int) and _limit > 0:
                _set = torch.utils.data.Subset(_set, torch.arange(_limit))
            
            self.bs = bs
            _loader = torch.utils.data.DataLoader(
                _set,
                batch_size=bs,
                shuffle=bool(shuffle) and _split=='train',
                num_workers=num_workers,
            )
            _batch_count = len(_loader)
            _sample_count = len(_set)
            self.sets[_split] = _set
            self.loaders[_split] = _loader
            self.info['batch_count'][_split] = _batch_count
            self.info['sample_count'][_split] = _sample_count


# %%
class CustomFrozenRepresentations(torch.utils.data.Dataset):
    """Custom frozen image representations dataset."""

    def __init__(self,
                reprs=None,
                labels=None,
                # transform=None,
                **kwargs):
        """
        Args:
        """
        # self.transform = transform
        self.reprs = reprs
        self.labels = labels
        
    def __len__(self):
        return len(self.reprs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        _repr = self.reprs[idx]
        _label = self.labels[idx]
        sample = [_repr, _label]
        
        return sample

# %%
class CustomFrozenDatasets:
    def __init__(self,
                train_reprs=None,
                train_labels=None,
                test_reprs=None,
                test_labels=None,
                splits=['train', 'test'],
                limit=0,
                num_workers=4,
                shuffle=False,
                bs=128,
                ):
        self.info = {
            'batch_count': {},
            'sample_count': {},
        }
        self.sets = {}
        self.loaders = {}
        for _split in splits:
            data_kwargs = {
                'train': {
                    'reprs': train_reprs,
                    'labels': train_labels,
                },
                'test': {
                    'reprs': test_reprs,
                    'labels': test_labels,
                },
            }
            _set = CustomFrozenRepresentations(
                **data_kwargs.get(_split, {})
            )
            if isinstance(limit, int) and limit > 0:
                _set = torch.utils.data.Subset(_set, torch.arange(limit))
            
            self.bs = bs
            _loader = torch.utils.data.DataLoader(
                _set,
                batch_size=bs,
                shuffle=bool(shuffle) and _split=='train',
                num_workers=num_workers,
            )
            _batch_count = len(_loader)
            _sample_count = len(_set)
            self.sets[_split] = _set
            self.loaders[_split] = _loader
            self.info['batch_count'][_split] = _batch_count
            self.info['sample_count'][_split] = _sample_count


# %%
class Datasets_Single:
    def __init__(self, num_labels=1000, splits=['train', 'val']):
        pass
    #     self.num_labels = num_labels
    #     self.dataset = None
    #     self.splits = splits
    
    # @property
    # def info(self):
    #     return {
    #         'dataset': self.dataset,
    #         'batch_count': {
    #             _split: None if self.batch_count is None else len(self.batch_count)[_split]
    #             for _split in self.splits
    #         },
    #         'sample_count': {
    #             _split: None if self.batch_count is None else len(self.batch_count)[_split]
    #             for _split in self.splits
    #         },
    #         'num_labels': self.num_labels,
    #     }
    
    @classmethod
    def get_trans(cls, image_size=32, resize=True, base_train_trans=True, auto_policy=None, norm_values=None, to_tensor=True):
        trans = {
            'train': [],
            'test': [],
        }
        
        if resize:
            trans['train'].append(transforms.Resize(image_size, F.InterpolationMode.BICUBIC))
            trans['test'].append(transforms.Resize(image_size, F.InterpolationMode.BICUBIC))
        
        if base_train_trans:
            trans['train'].extend([
                transforms.RandomCrop(image_size, padding=max(2, int(image_size//12)), fill=128),
                transforms.RandomHorizontalFlip(),
            ])
        if auto_policy is not None:
            if auto_policy is True:
                auto_policy = ImageNetPolicy()
            trans['train'].append(auto_policy)
        
        if to_tensor:
            trans['train'].append(transforms.ToTensor())
            trans['test'].append(transforms.ToTensor())
        
        if isinstance(norm_values, dict):
            trans['train'].append(transforms.Normalize(**norm_values))
            trans['test'].append(transforms.Normalize(**norm_values))
        
        return trans

# %%
class Datasets_STL10(Datasets_Single):
    norm_values = {
        'mean': [0.44671062065972217, 0.43980983983523964, 0.40664644709967324],
        'std': [0.2603409782662331, 0.25657727311344447, 0.27126738145225493],
    }
    num_labels = 10
    image_shape = [96, 96, 3]
    image_size = 96
    
    def __init__(self):
        pass
    
    @classmethod
    def get_sets(cls,
                image_size=0,
                root_path='/host/ubuntu/torch',
                norm_values=True,
                to_tensor=True,
                auto_policy=False,
                transform_pre=[],
                transform_post=[],
                base_train_trans=True,
                ):
        
        if not isinstance(image_size, int) or image_size <= 0:
            image_size = cls.image_size
        
        if norm_values is True:
            norm_values = cls.norm_values
        
        trans = cls.get_trans(
            image_size=image_size,
            resize=image_size != cls.image_size,
            base_train_trans=base_train_trans,
            auto_policy=STL10Policy() if auto_policy else None,
            norm_values=norm_values,
            to_tensor=to_tensor,
        )
        
        _sets = {}
        for _split, _training in zip(['train', 'test'], [True, False]):
            _sets[_split] = torchvision.datasets.STL10(
                root=root_path,
                split=_split,
                # folds=None,
                transform=transforms.Compose([
                    *transform_pre,
                    *trans[_split],
                    *transform_post,
                ]),
                # target_transform=None,
                download=True,
            )
        return _sets



# %%
class Datasets_CIFAR10(Datasets_Single):
    norm_values = {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.247, 0.243, 0.261],
    }
    num_labels = 10
    image_shape = [32, 32, 3]
    image_size = 32
    
    def __init__(self):
        pass
    
    @classmethod
    def get_sets(cls,
                image_size=0,
                root_path='/host/ubuntu/torch',
                norm_values=True,
                to_tensor=True,
                auto_policy=False,
                transform_pre=[],
                transform_post=[],
                base_train_trans=True,
                ):
        
        if not isinstance(image_size, int) or image_size <= 0:
            image_size = cls.image_size
        
        if norm_values is True:
            norm_values = cls.norm_values
        
        trans = cls.get_trans(
            image_size=image_size,
            resize=image_size != cls.image_size,
            base_train_trans=base_train_trans,
            auto_policy=CIFAR10Policy() if auto_policy else None,
            norm_values=norm_values,
            to_tensor=to_tensor,
        )
        
        _sets = {}
        for _split, _training in zip(['train', 'test'], [True, False]):
            _sets[_split] = torchvision.datasets.CIFAR10(
                root=root_path,
                train=_training,
                transform=transforms.Compose([
                    *transform_pre,
                    *trans[_split],
                    *transform_post,
                ]),
                download=True,
            )
        return _sets


# %%
# 100

class Datasets_CIFAR100(Datasets_Single):
    norm_values = {
        'mean': [0.50707516,  0.48654887,  0.44091784],
        'std': [0.26733429,  0.25643846,  0.27615047],
    }
    num_labels = 100
    image_shape = [32, 32, 3]
    image_size = 32
    
    def __init__(self):
        pass
    
    @classmethod
    def get_sets(cls,
                image_size=0,
                root_path='/host/ubuntu/torch',
                norm_values=True,
                to_tensor=True,
                auto_policy=False,
                transform_pre=[],
                transform_post=[],
                base_train_trans=True,
                ):
        
        if not isinstance(image_size, int) or image_size <= 0:
            image_size = cls.image_size
        
        if norm_values is True:
            norm_values = cls.norm_values
        
        trans = cls.get_trans(
            image_size=image_size,
            resize=image_size != cls.image_size,
            base_train_trans=base_train_trans,
            auto_policy=CIFAR10Policy() if auto_policy else None,
            norm_values=norm_values,
            to_tensor=to_tensor,
        )
        
        _sets = {}
        for _split, _training in zip(['train', 'test'], [True, False]):
            _sets[_split] = torchvision.datasets.CIFAR10(
                root=root_path,
                train=_training,
                transform=transforms.Compose([
                    *transform_pre,
                    *trans[_split],
                    *transform_post,
                ]),
                download=True,
            )
        return _sets





# %%
class Datasets:
    available_datasets = [
        'stl10',
        'cifar10',
        'cifar100',
    ]
    def __init__(self,
                dataset='stl10',
                root_path='/tmp',
                batchsize=128,
                transform_pre=[],
                transform_post=[],
                download=True,
                shuffle=True,
                num_workers=4,
                limit_train=0,
                limit_test=0,
                ds_kwargs={},
                resize=0,
                ddp=None,
                image_size=0,
                splits=['train', 'test'],
                ):
        
        
        if dataset == 'stl10':
            _sets = Datasets_STL10.get_sets(
                image_size=image_size,
                root_path=root_path,
                norm_values=True,
                to_tensor=True,
                auto_policy=False,
                transform_pre=transform_pre,
                transform_post=transform_post,
                base_train_trans=True,
            )
            _num_labels = Datasets_STL10.num_labels
        elif dataset == 'cifar10':
            _sets = Datasets_CIFAR10.get_sets(
                image_size=image_size,
                root_path=root_path,
                norm_values=True,
                to_tensor=True,
                auto_policy=False,
                transform_pre=transform_pre,
                transform_post=transform_post,
                base_train_trans=True,
            )
            _num_labels = Datasets_CIFAR10.num_labels
        elif dataset == 'cifar100':
            _sets = Datasets_CIFAR100.get_sets(
                image_size=image_size,
                root_path=root_path,
                norm_values=True,
                to_tensor=True,
                auto_policy=False,
                transform_pre=transform_pre,
                transform_post=transform_post,
                base_train_trans=True,
            )
            _num_labels = Datasets_CIFAR100.num_labels
        else:
            raise ValueError('dataset [{}] is not supported'.format(dataset))
        
        # splits = ['train', 'test']
        self.data = {
            _split: {
                'set_full': _sets[_split],
                'set': _sets[_split],
                'loader': None,
                'limit': [limit_train, limit_test][i],
                'batch_count': None,
                'sample_count': None,
                'bs': 1,
            }
            for i, _split in enumerate(splits)
        }
        
        self.num_labels = _num_labels
        self.dataset = str(dataset)
        self.root_path = root_path
        
        self.bs = batchsize
        if isinstance(self.bs, int):
            self.bs = [self.bs for _ in range(len(splits))]
        if isinstance(self.bs, (list, tuple)):
            assert len(self.bs) == len(splits)
            for v, _split in zip(self.bs, splits):
                self.data[_split]['bs'] = v
            self.bs = {
                k: v
                for v, k in zip(self.bs, splits)
            }
        else:
            raise ValueError('datasets batchsize')
        
        self.sets = _sets
        self.loaders = {}
        self.info = {
            'dataset': self.dataset,
            'batch_count': {},
            'sample_count': {},
            'num_labels': self.num_labels,
        }
        # self.limits = {
        #     'train': limit_train,
        #     'test': limit_test,
        # }
        self.ddp = ddp
        for i, _split in enumerate(splits):
            _set = self.data[_split]['set_full']
            _limit = self.data[_split]['limit']
            _bs = self.data[_split]['bs']
            
            if isinstance(_limit, int) and _limit > 0:
                _set = torch.utils.data.Subset(_set, torch.arange(_limit))
            
            _loader = None
            if ddp is not None:
                # set up for distributed learning
                _sampler = torch.utils.data.DistributedSampler(
                    _set,
                    num_replicas=ddp['size'],
                    rank=ddp['rank'],
                    shuffle=shuffle and _split == 'train',
                )
                _loader = torch.utils.data.DataLoader(
                    _set,
                    sampler=_sampler,
                    batch_size=_bs,
                    num_workers=num_workers,
                    pin_memory=False,
                    drop_last=False,
                )
            else:
                _loader = torch.utils.data.DataLoader(
                    _set,
                    batch_size=_bs,
                    shuffle=bool(shuffle) and _split == 'train',
                    num_workers=num_workers,
                )
            self.data[_split]['batch_count'] = len(_loader)
            self.data[_split]['sample_count'] = len(_set)
            self.data[_split]['set'] = _set
            self.data[_split]['loader'] = _loader
            
            self.sets[_split] = self.data[_split]['set']
            self.loaders[_split] = self.data[_split]['loader']
            self.info['batch_count'][_split] = self.data[_split]['batch_count']
            self.info['sample_count'][_split] = self.data[_split]['sample_count']


# %%
class LocalDatasets:
    def __init__(self,
                dataset='tire',
                path='/host/ubuntu/torch/tire/tire_500',
                batchsize=128,
                transform_fns=[],
                transform_fns_train=[],
                transform_fns_test=[],
                transform_fns_post=[
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
                ],
                shuffle=True,
                num_workers=4,
                # splits=['train', 'test'],
                train_ratio=0.8,
                limit=0,
                # limit_train=0,
                # limit_test=0,
                num_labels=5,
                # ds_kwargs={},
                # ddp=None,
                ):
        splits = ['train', 'test']
        self.dataset = dataset
        self.num_labels = int(num_labels)
        self.bs = batchsize
        self.sets = {}
        self.loaders = {}
        self.info = {
            'dataset': self.dataset,
            'batch_count': {},
            'sample_count': {},
            'num_labels': self.num_labels,
        }
        # self.limits = {
        #     'train': limit_train,
        #     'test': limit_test,
        # }
        
        
        norm_values = {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.25, 0.25, 0.25],
        }
        transform_dict = {
            k: transforms.Compose([
                *transform_fns,
                *fns,
                *transform_fns_post,
                # transforms.Resize(255),
                # transforms.CenterCrop(224),
            ])
            for k, fns in zip(['train', 'test'], [transform_fns_train, transform_fns_test])
        }
        self.path = path
        assert os.path.isdir(self.path)
        ds_full = {
            _split: torchvision.datasets.ImageFolder(
                self.path,
                transform=transform_dict[_split],
            )
            for _split in splits
        }
        ds = {}
        for _split in splits:
            ds[_split] = ds_full[_split]
        loader = torch.utils.data.DataLoader(
            ds_full['train'],
            batch_size=1,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )

        # print('found {} images, onto {} batches'.format(len(ds), len(loader)))
        # _shapes = []
        
        _labels = []
        for i, data in enumerate(loader):
            images, labels = data
            # _shapes.append(list(images.shape))
            _labels_new = list(labels.tolist())
            _labels.extend(_labels_new)
        
        _total_count = len(_labels)
        _limit_ratio = 1.0
        if isinstance(limit, int) and limit > 0:
            _limit_ratio = min(1.0, limit / _total_count)
        
        # label_agg = {}
        label_idx = {}
        for i, _label in enumerate(_labels):
            if _label not in label_idx:
                label_idx[_label] = []
            label_idx[_label].append(i)
            # if _label not in label_agg:
            #     label_agg[_label] = 0
            # label_agg[_label] += 1

        label_idx
        
        label_idx_split = {
            'train': {},
            'test': {},
        }
        self.train_ratio = float(min(1, max(0, train_ratio)))
        self.count = {'train': 0, 'test': 0}
        for k, v in label_idx.items():
            _limited_len = len(v) * _limit_ratio
            _train_count = int(np.floor(_limited_len * self.train_ratio))
            _test_count = int(np.floor(_limited_len)) - _train_count
            
            # print('class [{}] count [{}] train [{}]'.format(k, len(v), _train_count))
            label_idx_split['train'][k] = v[ : _train_count]
            label_idx_split['test'][k] = v[_train_count: _train_count + _test_count]
            self.count['train'] += len(label_idx_split['train'][k])
            self.count['test'] += len(label_idx_split['test'][k])
        label_idx_split
        
        split_idx = {
            k: [
                v2
                for v1 in v.values()
                for v2 in v1
            ]
            for k, v in label_idx_split.items()
        }
        # np.array(split_idx['train'])
        # np.array(split_idx['val'])
        
        samplers = {
            k: torch.utils.data.SubsetRandomSampler(indices=v)
            for k, v in split_idx.items()
        }
        self.samplers = samplers
        
        loaders = {
            _split: torch.utils.data.DataLoader(
                ds[_split],
                sampler=_sampler,
                batch_size=self.bs,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
            )
            for _split, _sampler in samplers.items()
        }
        loaders
        
        for _split in splits:
            # if isinstance(limit, int) and limit > 0:
            #     _set = torch.utils.data.Subset(_set, torch.arange(limit))
            self.sets[_split] = ds[_split]
            self.loaders[_split] = loaders[_split]
            self.info['batch_count'][_split] = len(loaders[_split])
            self.info['sample_count'][_split] = self.count[_split]
    


# %%
class TRANS:
    
    lbp_methods = [
        'default',
        'ror',
        'uniform',
        'nri_uniform',
        # 'var',
    ]
    @classmethod
    def _lbp_uniform(cls, x):
        radius = 2
        n_points = 8 * radius
        METHOD = 'uniform'
        imgUMat = np.float32(x)
        gray = cv2.cvtColor(imgUMat, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        lbp = torch.from_numpy(lbp).float()
        return lbp
    
    @classmethod
    def _lbp_full3(cls, x, radius):
        n_points = 8 * radius
        METHOD = 'uniform'
        imgUMat = np.float32(x)
        gray = cv2.cvtColor(imgUMat, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, '')
        return lbp
    
    @classmethod
    def get_lbp_full(cls, img, radius=1, point_mult=8, methods=None):
        if isinstance(img, Image.Image):
            if img.mode != 'L':
                return cls.get_lbp_full(np.array(img.convert('L')), radius, point_mult, methods)
            return cls.get_lbp_full(np.array(img), radius, point_mult, methods)
        if not isinstance(img, np.ndarray):
            raise ValueError('`img` must be of type [ numpy.ndarray | PIL.Image.Image ]')
        if len(img.shape) != 2 or np.prod(img.shape) <= 0:
            return cls.get_lbp_full(Image.fromarray(img).convert('L'), radius, point_mult, methods)
            # raise ValueError('`img` must be a non-empty rank-2 numpy array (gray-scale image)')
        if methods is None:
            methods = [*cls.lbp_methods]
        if isinstance(methods, str):
            methods = [methods]
        _n_points = min(point_mult * radius, 24)
        r = {}
        for _method in methods:
            _range = [0, 255]
            if _method == 'default':
                _range = [0, 2 ** _n_points - 1]
            elif _method == 'ror':
                _range = [0, 2 ** _n_points - 1]
            elif _method == 'uniform':
                _range = [0, _n_points + 1]
            elif _method == 'nri_uniform':
                _range = [0, (_n_points + 1) * _n_points]
            # if _method == 'var':
            #     img_lbp = img_lbp / (2 ** _n_points) * 255
            # elif _method == 'original':
            #     r[_method] = img
            #     continue
            else:
                continue
                # raise NotImplementedError('method [{}] has not been implemented (internal error)'.format(_method))
            img_lbp = local_binary_pattern(img, _n_points, radius, _method)
            img_lbp = (img_lbp - _range[0]) / (_range[1] - _range[0]) * 255
            img_lbp = np.clip(img_lbp, 0, 255).astype(np.uint8)
            # print('default in [0~{})'.format(np.max(img_lbp)))
            # fig = px.histogram(img_lbp.reshape(-1)[np.all([img_lbp.reshape(-1) > 0, img_lbp.reshape(-1) < 255], axis=0)], nbins=256)
            # fig.show()
            r[_method] = img_lbp
        return r
    
    @classmethod
    def get_lbp_merge(cls, img, radius=1, point_mult=8, methods=['l', 'default', 'uniform']):
        assert isinstance(methods, (list, tuple))
        # assert len(methods) == 3
        assert all([v in [*cls.lbp_methods, 'l', 'r', 'g', 'b'] for v in methods])
        if isinstance(img, Image.Image):
            return cls.get_lbp_merge(np.array(img), radius, point_mult, methods)
        if not isinstance(img, np.ndarray):
            raise ValueError('`img` must be of type [ numpy.ndarray | PIL.Image.Image ]')
        # if len(img.shape) != 2 or np.prod(img.shape) <= 0:
        #     raise ValueError('`img` must be a non-empty rank-2 numpy array (gray-scale image)')
        # _img_np = np.array(img_raw_L.resize([1000, 1000], Image.BICUBIC))
        _channel = len(methods)
        _shape = list(img.shape)[:2]
        imgs_lbp = cls.get_lbp_full(img, radius, point_mult, methods=methods)
        
        img_merge_rgb = np.zeros([*_shape, _channel], np.uint8)
        for i, _method in enumerate(methods):
            if _method in cls.lbp_methods:
                img_merge_rgb[:, :, i] = imgs_lbp[_method]
            else:
                if len(img.shape) == 3:
                    assert img.shape[2] == 3
                    # img is RGB
                    if _method in ['l']:
                        img_merge_rgb[:, :, i] = np.array(Image.fromarray(img).convert('L'))
                    
                    for j, c in enumerate(['r', 'g', 'b']):
                        if _method == c:
                            img_merge_rgb[:, :, i] = np.array(img[:, :, j])
                            break
                elif len(img.shape) == 2:
                    # img is L
                    if _method in ['l', 'r', 'g', 'b']:
                        img_merge_rgb[:, :, i] = np.array(img)
                else:
                    raise ValueError('image shape [{}] not supported'.format(img.shape))
        
        # Image.fromarray(img_merge_rgb)
        return img_merge_rgb
    
    @classmethod
    def lbp_merge(cls, radius=1, point_mult=8, methods=['l', 'default', 'uniform']):
        return transforms.Lambda(lambda img: cls.get_lbp_merge(
            img,
            radius=radius,
            point_mult=point_mult,
            methods=methods,
        ))
        # return transforms.Lambda(lambda img: Image.fromarray(cls.get_lbp_merge(
        #     img,
        #     radius=radius,
        #     point_mult=point_mult,
        #     methods=methods,
        # )))
    
    
    @classmethod
    def get_fit_to(cls, img, shape=1000, fill=0, interpolation=Image.BICUBIC):
        if isinstance(shape, int):
            assert shape > 0
            shape = [shape, shape]
        _size = img.size
        # assert len(_size) in [2, 3]
        # _channel = None
        _shape = _size[:2]
        # print(img.size, shape, fill)
        if img.mode in ['RGB', 'L']:
            img2 = Image.new(img.mode, shape, fill)
        else:
            raise ValueError('img must have be rank 2 or 3')
        shape_r = [int(v * v0 / max(_shape)) for v, v0 in zip(shape, _shape)]
        img_r = img.resize(shape_r, resample=interpolation)
        img2.paste(
            img_r,
            [
                int(np.ceil((a - b) / 2))
                for a, b in zip(shape, list(shape_r))
            ],
        )
        return img2
    
    @classmethod
    def fit_to(cls, shape=1000, fill=0, interpolation=Image.BICUBIC):
        return transforms.Lambda(lambda img: cls.get_fit_to(
            img,
            shape=shape,
            fill=fill,
            interpolation=interpolation,
        ))
    
    @classmethod
    def get_pad_to(cls, img, shape=1000, fill=0):
        if isinstance(shape, int):
            assert shape > 0
            shape = [shape, shape]
        _size = img.size
        assert len(_size) in [2, 3]
        _channel = None
        _shape = _size[:2]
        if len(_size) >= 3:
            # _channel = _size[2]
            # assert _channel == 3
            img2 = Image.new('RGB', _shape, fill)
        else:
            img2 = Image.new('L', _shape, fill)
        img2.paste(
            img,
            [
                int(np.ceil((a - b) / 2))
                for a, b in zip(shape, list(_shape))
            ],
        )
        return img2
    
    @classmethod
    def pad_to(cls, shape=1000, fill=0):
        return transforms.Lambda(lambda img: cls.get_pad_to(
            img,
            shape=shape,
            fill=fill,
        ))


class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')



# %%
if __name__ == '__main__':
    ds = Datasets(
        dataset='cifar10',
        root_path='/host/ubuntu/torch',
        batchsize=[128, 128],
        transform=[],
        download=True,
        splits=['train', 'test'],
        shuffle=True,
        num_workers=4,
    )
    d = list(ds.loaders['train'])
    d[0][0].shape, d[0][0].mean()
    ds.info


# %%