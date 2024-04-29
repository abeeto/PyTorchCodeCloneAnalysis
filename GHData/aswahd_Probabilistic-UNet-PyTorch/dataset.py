import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from pathlib import Path
import mimetypes
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt






IMAGE_EXTENSIONS = [k.lower() for k, v in mimetypes.types_map.items() if v.startswith('image')]


def collate_fn(data):
    """
    :param data: a list tuples with (img, semantic_mask, instance_mask)
    img: [3, H, W]
    semantic_mask: [1, H, W]
    instance_mask: [n_instances, H, W]
    :return:
    """

    img, semantic_mask, instance_mask = zip(*data)

    # If the mask doesn't contain any foreground instances,
    # the number of instances will be 0.
    n_instances = []
    for mask in instance_mask:
        if mask.shape[0] == 1:
            no_instance = lambda x: x.eq(0).all()
            if no_instance(mask):
                n_instances.append(0)
            else:
                n_instances.append(1)
        else:
            n_instances.append(mask.shape[0])
    n_instances = torch.tensor(n_instances).long()

    # Pad zeros
    max_depth = max([mask.shape[0] for mask in instance_mask])
    _, H, W = img[0].shape
    padded_instance_mask = []
    for mask in instance_mask:
        amount_to_pad = max_depth - mask.shape[0]
        padding = torch.zeros(amount_to_pad, H, W)
        padded_instance_mask.append(torch.cat([mask, padding], dim=0))

    return torch.stack(img), torch.stack(semantic_mask), torch.stack(padded_instance_mask), n_instances


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), 1
        return img, 0


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.vflip(img), 1
        return img, 0


class RandomRotation(torch.nn.Module):

    def __init__(self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None):
        super().__init__()

        self.resample = self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.expand = expand
        self.degrees = float(degrees)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())

        return F.rotate(img, angle, self.resample, self.expand, self.center, 0), (angle, self.resample, self.expand, self.center, 0)


class RandomCrop(torch.nn.Module):

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = (size, size) if isinstance(size, int) else size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = img.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), (i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class LeafDataset(Dataset):

    def __init__(self, root):
        self.root = Path(root)
        path = self.root / 'training_data_challenge/A1'
        image_file_names = [fn for fn in list(path.iterdir()) if fn.is_file()
                            and '_rgb' in fn.stem
                            and fn.suffix.lower() in IMAGE_EXTENSIONS]
        mask_file_names = [fn for fn in list(path.iterdir()) if fn.is_file()
                           and '_label' in fn.stem
                           and fn.suffix.lower() in IMAGE_EXTENSIONS]

        self.x = image_file_names
        self.y = mask_file_names

        # transforms
        # 1: Random horizontal flip
        # 2: Random vertical flip
        # 3: Random rotation
        # 5. Normalize
        # 6: Random crop: (512, 512)
        # 7. Resize: (300, 400)

        self.horizontal_flip = RandomHorizontalFlip(0.2)
        self.vertical_flip = RandomVerticalFlip(0.2)
        self.random_rotation = RandomRotation(20)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5198, 0.3845, 0.2066), (0.2126, 0.1491, 0.1108))
        self.resize = transforms.Resize((256, 256), F.InterpolationMode.NEAREST)

    def __getitem__(self, item):
        # Read image and mask
        img = Image.open(self.x[item]).convert('RGB')
        mask = Image.open(self.y[item]).convert('L')

        # Random horizontal flip
        # img, flipped = self.horizontal_flip(img)
        # if flipped:
        #     mask = F.hflip(mask)
        #
        # # Random vertical flip
        # img, flipped = self.vertical_flip(img)
        # if flipped:
        #     mask = F.vflip(mask)
        #
        # # Random rotation
        # img, params = self.random_rotation(img)
        # mask = F.rotate(mask, *params[:-1], 0)  # Fill with background pixels

        # Resize
        img = self.resize(img)
        mask = self.resize(mask)

        # to tensor
        img = self.to_tensor(img)

        # Normalize
        img = self.normalize(img)

        mask = np.array(mask)
        labels = np.unique(mask)
        labels = np.delete(labels, np.where(labels == 0))  # remove background label

        mask = torch.from_numpy(mask)
        instance_mask = []

        for label in labels:
            m = torch.zeros_like(mask)
            m[mask == label] = 1
            instance_mask.append(m)

        if len(labels) > 0:
            instance_mask = torch.stack(instance_mask).float()
        else:
            # if image contains no instances
            instance_mask = torch.zeros_like(mask[None], dtype=torch.float32)

        return img, instance_mask


    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    # Test
    ds = LeafDataset('CVPPP2014_LSC_training_data')
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    img, inst_mask = next(iter(dataloader))
    print(img.shape, inst_mask.shape)

