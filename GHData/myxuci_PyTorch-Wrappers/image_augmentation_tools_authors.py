import numbers

import torchvision.transforms.functional as F
from torchvision.transforms import transforms

# Define normalization parameters:
normalize_torch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize_05 = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def preprocess(normalize, image_size):
    assert isinstance(image_size, int), "Parameter image size can only be int."
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_hflip(normalize, image_size):
    assert isinstance(image_size, int), "Parameter image size can only be int."
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])

class HorizontalFlip(object):
    """Horizontally flip the given PIL Image."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Flipped image.
        """
        return F.hflip(img)


def five_crop(img, size, crop_pos):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    if crop_pos == 0:
        return img.crop((0, 0, crop_w, crop_h))
    elif crop_pos == 1:
        return img.crop((w - crop_w, 0, w, crop_h))
    elif crop_pos == 2:
        return img.crop((0, h - crop_h, crop_w, h))
    elif crop_pos == 3:
        return img.crop((w - crop_w, h - crop_h, w, h))
    else:
        return F.center_crop(img, (crop_h, crop_w))


class FiveCropParametrized(object):
    def __init__(self, size, crop_pos):
        self.size = size
        self.crop_pos = crop_pos
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return five_crop(img, self.size, self.crop_pos)


def five_crops(size):
    return [FiveCropParametrized(size, i) for i in range(5)]


def make_transforms(first_part, second_part, inners):
    return [transforms.Compose(first_part + [inner] + second_part) for inner in inners]