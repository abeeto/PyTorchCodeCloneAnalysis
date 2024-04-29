import numpy as np
import torch

max_bit16 = (2**16) - 1

'''
if False:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import matplotlib.pyplot as plt
    plt.switch_backend('tkagg')
    sys.path.append('/home/assaf/workspace/noise_flow')
    from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
    
    class MyNoiseFlowWrapper:
        def __init__(self, nf_model_path = '/home/assaf/workspace/noise_flow/models/NoiseFlow',
                     patch_size=32, stride=32, cam=0, iso=100.0, b1=-10, b2=-3):
            self.model = NoiseFlowWrapper(nf_model_path)
            self.patch_size = patch_size
            self.stride = stride
            self.iso = iso
            self.cam = cam
            self.b1 = b1
            self.b2 = b2

        def batch_to_patch_batch(self, clean_batch):
            """ in: b,c,h,w - c: rgb
                out: b,h,w,c - c: rggb"""
            b, c, h, w = clean_batch.shape
            patches = []
            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    patch_4C = np.zeros((b, self.patch_size, self.patch_size, 4), dtype=np.float32)
                    patch_3C = clean_batch[:, :, i:i + self.patch_size, j:j + self.patch_size].transpose(0, 2, 3, 1)
                    patch_4C[:, :, :, :2] = patch_3C[:, :, :, 2:]
                    patch_4C[:, :, :, 2] = patch_3C[:, :, :, 1]
                    patch_4C[:, :, :, 3] = patch_3C[:, :, :, 2]
                    patches.append(patch_4C)
            batch = np.vstack(patches)
            return batch

        def add_noise(self, noise_batch, clean_batch):
            b, c, h, w = clean_batch.shape
            patch_id = 0
            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    patch = noise_batch[patch_id:patch_id + b].transpose(0, 3, 1, 2)
                    clean_batch[:, 0, i:i + self.patch_size, j:j + self.patch_size] += patch[:, 0] / 10
                    clean_batch[:, 1, i:i + self.patch_size, j:j + self.patch_size] += patch[:, 1] / 10
                    clean_batch[:, 2, i:i + self.patch_size, j:j + self.patch_size] += patch[:, 3] / 40
                    patch_id += b
            np.clip(clean_batch, 0, 1, out=clean_batch)

        def make_noise(self, clean_batch):
            batch = self.batch_to_patch_batch(clean_batch)
            noise_batch = self.model.sample_noise_nf(batch, self.b1, self.b2, self.iso, self.cam)
            self.add_noise(noise_batch, clean_batch)

        def nf_collate_fn_random_rot90(self, data):
            """Creates mini-batch tensors from the list of tuples (image, caption)."""
            images, labels = zip(*data)
            # Merge arrays (from tuple of 3D to 4D).
            images = np.stack(images, 0)
            # import cv2
            # cv2.imwrite('/tmp/1.png', cv2.cvtColor((images[0].transpose(1, 2, 0) * 255).astype(np.uint8),
                          cv2.COLOR_RGB2BGR))
            self.make_noise(images)
            # cv2.imwrite('/tmp/2.png', cv2.cvtColor((images[0].transpose(1, 2, 0) * 255).astype(np.uint8),
                          cv2.COLOR_RGB2BGR))
            labels = np.stack(labels, 0)
            if np.random.rand() > 0.5:
                images = np.rot90(images, axes=(2, 3))
                if labels.ndim > 3:
                    labels = np.rot90(labels, axes=(2, 3))
                else:
                    labels = np.rot90(labels, axes=(1, 2))
            im_tensor = torch.as_tensor(data=np.ascontiguousarray(images), dtype=torch.float32)
            lbl_tensor = torch.as_tensor(data=np.ascontiguousarray(labels), dtype=torch.float32)
            return im_tensor, lbl_tensor
'''


def pad_2d(img: np.ndarray, divisor, dims='hwc') -> np.ndarray:
    """return padded image to shape of nearest divisor"""
    if np.sum(np.array(img.shape[:2]) % divisor) == 0:
        return img
    expand = divisor - (np.array(img.shape[:2]) % divisor)
    expand = np.bitwise_xor(expand, divisor)
    top, left = expand // 2
    bottom, right = expand - (expand // 2)
    if img.ndim > 2:
        return np.pad(img, pad_width=((top, bottom), (left, right), (0, 0)), mode='symmetric')
    else:
        return np.pad(img, pad_width=((top, bottom), (left, right)), mode='symmetric')


def random_crop(img, label, patch_size, edge_border):
    height, width = img.shape[:2]
    h_c = int(np.random.rand() * (height - (edge_border * 2) - patch_size)) + edge_border
    w_c = int(np.random.rand() * (width - (edge_border * 2) - patch_size)) + edge_border
    img = img[h_c:h_c + patch_size, w_c:w_c + patch_size]
    label = label[h_c:h_c + patch_size, w_c:w_c + patch_size]
    return img, label


def random_dual_augmentation(image, label, sigma, pad_divisor=0, img_min=0, im_max=1, align_to_tensor=True,
                             augmentations=(np.flipud, np.fliplr), do_transpose=False):
    """torch transformations cannot be applied to a pair of images, so I've decided to implement in numpy."""
    if sigma > 0:
        gauss = np.random.normal(0, sigma, image.shape).astype(image.dtype)
        image = image + gauss
        image = np.clip(image, img_min, im_max)

    for augment in augmentations:
        if np.random.rand() > 0.5:
            image = augment(image)
            label = augment(label)
    if align_to_tensor:
        if pad_divisor > 0:
            image = pad_2d(image, pad_divisor)
            label = pad_2d(label, pad_divisor)
        if image.ndim > 2 and do_transpose:
            image = np.transpose(image, (2, 0, 1))
        if label.ndim > 2 and do_transpose:
            label = np.transpose(label, (2, 0, 1))
    return image, label


def base_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""
    images, labels = zip(*data)
    # Merge arrays (from tuple of 3D to 4D).
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    im_tensor = torch.as_tensor(data=images.astype(np.float32) / max_bit16, dtype=torch.float32)
    lbl_tensor = torch.as_tensor(data=labels.astype(np.float32) / max_bit16, dtype=torch.float32)
    if im_tensor.ndim < lbl_tensor.ndim:
        im_tensor = im_tensor.unsqueeze(1)
    return im_tensor, lbl_tensor


def base_collate_fn_random_rot90(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""
    images, labels = zip(*data)
    # Merge arrays (from tuple of 3D to 4D).
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    if np.random.rand() > 0.5:
        images = np.rot90(images, axes=(2, 3) if images.ndim > 3 else (1, 2))
        labels = np.rot90(labels, axes=(2, 3) if labels.ndim > 3 else (1, 2))
    return images, labels


def coll_fn_rand_rot90_float_long(data):
    images, labels = base_collate_fn_random_rot90(data)
    im_tensor = torch.as_tensor(data=images, dtype=torch.float32)
    lbl_tensor = torch.as_tensor(data=labels, dtype=torch.long)
    return im_tensor, lbl_tensor


def coll_fn_rand_rot90_float(data):
    images, labels = base_collate_fn_random_rot90(data)
    if images.ndim < 4:
        images = np.expand_dims(images, 1)
    if labels.ndim < 4:
        labels = np.expand_dims(labels, 1)
    images = images.astype(np.float32) / max_bit16
    labels = labels.astype(np.float32) / max_bit16
    im_tensor = torch.as_tensor(data=images, dtype=torch.float32)
    lbl_tensor = torch.as_tensor(data=labels, dtype=torch.float32)
    return im_tensor, lbl_tensor


def mosaic_image(rgb, pattern='rggb'):
    if pattern == 'rggb':
        rggb = np.zeros(rgb.shape[:2], dtype=rgb.dtype)
        rggb[::2, ::2] = rgb[::2, ::2, 0]    # r
        rggb[1::2, ::2] = rgb[1::2, ::2, 1]  # g1
        rggb[::2, 1::2] = rgb[::2, 1::2, 1]  # g2
        rggb[1::2, 1::2] = rgb[1::2, 1::2, 2]   # b
        return rggb


def myplot(im, ref):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.transpose(im, (1, 2, 0)))
    ax2.imshow(ref)
    plt.show()


def num_of_channels(image: np.ndarray):
    """returns the number of channels in image"""
    if image.ndim == 2:
        return 1
    else:
        return image.shape[2]
