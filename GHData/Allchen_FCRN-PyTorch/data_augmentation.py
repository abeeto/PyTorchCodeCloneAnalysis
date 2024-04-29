import numpy as np
import numpy.random as random
import torchvision.transforms as transforms
from transforms_cv import *
from transform_mutual import TransformMutual


def transform_rgb(rgb, mutual):
    if not (len(rgb.shape) == 3 and rgb.shape[2] == 3):
        msg = 'This function only support '
        msg += 'numpy array with 3 channels'
        raise TypeError(msg)
        return rgb

    rgb = rgb / 255
    # Perform scale, rotation and flip transformation on OpenCV image(numpy).
    if mutual.flip:
        rgb = hflip_transform(rgb)
    rgb = rescale_transform(rgb, (mutual.rescale, mutual.rescale))
    rgb = rotation_transform(rgb, mutual.rotate)
    rgb = crop_transform(
                    rgb, mutual.translation_x1,
                    mutual.translation_y1,
                    mutual.translation_x2,
                    mutual.translation_y2,
                    )
    rgb = cv2.resize(
        rgb, (mutual.output_size[1], mutual.output_size[0]),
        rgb, 0, 0,
        interpolation=cv2.INTER_LINEAR
        )

    # Convert OpenCV image into PyTorch Tensor
    rgb = np.transpose(rgb, [2, 0, 1])
    color_factor = random.rand(3) * 0.4 + np.ones(3) * 0.8
    rgb = torch.from_numpy(rgb).float()
    rgb_disp = torch.Tensor(3, mutual.output_size[0], mutual.output_size[1])
    rgb_disp = rgb_disp.copy_(rgb)

    # Perform random color transformation on PyTorch Tensor.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean /= color_factor
    std /= color_factor
    transform_final = transforms.Compose([
        transforms.Normalize(mean, std)
        ])
    rgb = transform_final(rgb)

    return rgb, rgb_disp


def transform_depth(depth, mutual):
    depth = depth.numpy()
    depth = depth / 10

    # Perform scale, rotation and flip transformation on OpenCV image.
    if mutual.flip:
        depth = hflip_transform(depth)
    depth = rescale_transform(depth, (mutual.rescale, mutual.rescale))
    depth /= mutual.rescale
    depth = rotation_transform(depth, mutual.rotate)
    depth = crop_transform(
                    depth, mutual.translation_x1,
                    mutual.translation_y1,
                    mutual.translation_x2,
                    mutual.translation_y2,
                    )
    depth = cv2.resize(
        depth, (mutual.output_size[1], mutual.output_size[0]),
        depth, 0, 0,
        interpolation=cv2.INTER_LINEAR
        )

    # Convert depth map into PyTorch Tensor
    depth = torch.from_numpy(depth).float()
    depth = depth.unsqueeze(0)
    return depth


def test():
    x = cv2.imread('mio.jpg')
    xd = cv2.imread('mio.jpg', 0)
    transform_mutual = TransformMutual(
                        img_size=(480, 640),
                        output_size=(480, 640),
                        rescale_range=(1, 1.5)
                        )

    y = transform_rgb(x, transform_mutual)
    y = y.squeeze()
    y = (y.numpy() * 255).astype(int)
    y = np.transpose(y, [1, 2, 0])

    yd = transform_depth(xd, transform_mutual)
    yd = yd.squeeze()
    yd = (yd.numpy() * 255).astype(int)

    cv2.imwrite('pro.jpg', y)
    cv2.imwrite('depth.jpg', yd)
    return


if __name__ == '__main__':
    test()
