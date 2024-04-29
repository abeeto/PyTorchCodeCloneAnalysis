import torch
import cv2
import PIL
from PIL import ImageEnhance, Image, ImageFilter
import PIL.ImageOps
import numpy as np
randint = np.random.randint




def get_array_color_mode(x):  # 获取图片的格式类型（灰度图还是彩色图）
    x = x.squeeze()
    if x.ndim == 2:
        mode = "L"
    elif x.ndim == 3 and x.shape[2] == 1:
        mode = "L"
        x = x.squeeze()
    elif x.ndim == 3:
        mode = "RGB"
    else:
        assert False, "Incapable of interpreting array as an image"

    return mode

def pil2array(im):  # 将PIL 图像转化为numpy矩阵
    return np.asarray(im, dtype=np.uint8)


def array2pil(x):  # 将numpy矩阵转化为PIL图像
    x = x.squeeze()
    return PIL.Image.fromarray(x.astype('uint8')).convert(get_array_color_mode(x))

class Resize(object):
    def __init__(self, height, width):
        self.width = width
        self.height = height

    def __call__(self, img, lbl):
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)[..., ::-1]
        lbl = cv2.resize(lbl, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return [img, lbl]


class random_crop(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, min_scale=0.5, max_scale=1.0, preserve_size=True, resample=cv2.INTER_NEAREST):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.preserve_size = preserve_size
        self.resample = resample

    def __call__(self, image, label):
        width, height = image.shape[:2]
        crop_width = np.random.randint(width * self.min_scale, width * self.max_scale)
        crop_height = np.random.randint(height * self.min_scale, height * self.max_scale)
        x_offset = np.random.randint(0, width - crop_width + 1)
        y_offset = np.random.randint(0, height - crop_height + 1)
        im2 = image[x_offset:x_offset + crop_width, y_offset:y_offset + crop_height, :]
        lb2 = label[x_offset:x_offset + crop_width, y_offset:y_offset + crop_height]
        if self.preserve_size:
            im2 = cv2.resize(im2, (width, height), interpolation=self.resample)[..., ::-1]
            lb2 = cv2.resize(lb2, (width, height), interpolation=self.resample)

        return [im2, lb2]


class random_90_rotation(object):
    def __init__(self):
        self.methods = [1, 2, 3]

    def __call__(self, img, lbl):
        method = np.random.choice(self.methods)
        img = np.rot90(img, method)
        lbl = np.rot90(lbl, method)
        return [img, lbl]


class random_rotation(object):  # 随机的旋转
    """ Creates a new image which is rotated by a random amount between
        [-max, +max] inclusive.

    Args:
        im:              (PIL image)
        max:             (int) Max angle (in degrees in either direction).
        include_corners: (bool)
                If True, then the image canvas is expanded at first to
                fit the rotated corners, and then rescaled back to
                original image size.

                If False, then the original image canvas remains intact,
                and the corners of the rotated image that fall outside
                this box are clipped off.
    Returns:
        PIL image with random rotation applied.
    """

    def __init__(self, rotation=10, include_corners=True, resample=PIL.Image.NEAREST):
        self.rotation = rotation
        self.include_corners = include_corners
        self.resample = resample

    def __call__(self, img, lbl):
        original_dims = img.shape[:2]
        angle = randint(-self.rotation, self.rotation + 1)
        if angle != 0:
            img, lbl = array2pil(img), array2pil(lbl)
            img = img.rotate(angle, resample=self.resample, expand=self.include_corners)
            lbl = lbl.rotate(angle, resample=self.resample, expand=self.include_corners)
            if self.include_corners:
                img = img.resize(original_dims, resample=self.resample)
                lbl = lbl.resize(original_dims, resample=self.resample)
            img, lbl = pil2array(img), pil2array(lbl)
        return [img, lbl]


class random_flip(object):
    def __init__(self):
        self.methods = [0, 1, 2]

    def __call__(self, img, lbl):
        if np.random.choice(self.methods) == 1:
            img = cv2.flip(img, 0)[..., ::-1]  # horizontal flip
            lbl = cv2.flip(lbl, 0)  # horizontal flip
        elif np.random.choice(self.methods) == 2:
            img = cv2.flip(img, 1)[..., ::-1]  # horizontal flip
            lbl = cv2.flip(lbl, 1)  # horizontal flip
        return [img, lbl]


class random_invert(object):  # 图片的像素值P = 255 - P
    def __init__(self):
        pass

    def __call__(self, img, lbl):
        if np.random.choice([0, 1]) == 1:
            img = 255 - img
        return [img, lbl]


class random_brightness(object):  # 随机的亮度变化
    """Creates a new image which randomly adjusts the brightness of `im` by
       randomly sampling a brightness value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling brightness value.
        min:  (int or float) Clip contrast value to be no lower than this.
        max:  (int or float) Clip contrast value to be no higher than this.


    Returns:
        PIL image with brightness randomly adjusted.
    """

    def __init__(self, sd=0.5, min=0, max=20):
        self.sd = sd
        self.min = min
        self.max = max

    def __call__(self, img, lbl):
        brightness = np.clip(np.random.normal(loc=1, scale=self.sd), self.min, self.max)
        enhancer = ImageEnhance.Brightness(array2pil(img))
        img = enhancer.enhance(brightness)
        img = pil2array(img)[..., ::-1]
        return [img, lbl]


class random_contrast(object):
    """Creates a new image which randomly adjusts the contrast of `im` by
       randomly sampling a contrast value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling contrast value.
        min:  (int or float) Clip contrast value to be no lower than this.
        max:  (int or float) Clip contrast value to be no higher than this.

    Returns:
        PIL image with contrast randomly adjusted.
    """

    def __init__(self, sd=0.5, min=0, max=10):
        self.sd = sd
        self.min = min
        self.max = max

    def __call__(self, img, lbl):
        contrast = np.clip(np.random.normal(loc=1, scale=self.sd), self.min, self.max)
        enhancer = ImageEnhance.Contrast(array2pil(img))
        img = pil2array(enhancer.enhance(contrast))[..., ::-1]
        return [img, lbl]


class random_blur(object):
    def __init__(self, min=0, max=5):
        self.min = min
        self.max = max

    def __call__(self, img, lbl):
        blur_radius = randint(self.min, self.max + 1)
        if blur_radius != 0:
            img = array2pil(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            img = pil2array(img)[..., ::-1]
        return [img, lbl]


class random_noise(object):
    def __init__(self, sd=5):
        self.noise_sd = np.random.randint(0, sd)

    def __call__(self, img, lbl):
        if self.noise_sd > 0:
            noise = np.random.normal(loc=0, scale=self.noise_sd, size=img.shape)
            img = np.clip(img + noise, 0, 255)
            return [img, lbl]
        else:
            return [img, lbl]


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        for i in range(3):
            image[:, :, i] -= self.mean[i]
        for i in range(3):
            image[:, :, i] /= self.std[i]

        return [image, label]


class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''

    def __init__(self, scale=1):
        '''
        :param scale: ESPNet-C's output is 1/8th of original image size, so set this parameter accordingly
        '''
        self.scale = scale  # original images are 2048 x 1024

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)), interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2, 0, 1))

        image_tensor = torch.from_numpy(image)  # .div(255)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int))  # torch.from_numpy(label)

        return [image_tensor, label_tensor]


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
