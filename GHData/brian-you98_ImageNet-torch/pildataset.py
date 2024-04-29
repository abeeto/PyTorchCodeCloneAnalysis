import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


# 图片缩放，使用PIL
def letterbox(im, new_shape=(224, 224), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.size  # current shape [width, height]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[0], new_shape[1])
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = im.resize(new_unpad, Image.NEAREST)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    new_image = Image.new('RGB', (new_unpad[0] + left + right, new_unpad[1] + top + bottom), color)  # 生成灰色图像
    new_image.paste(im, (left, top))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image, ratio, (dw, dh)


# 使用PIL结合torchvision设置数据集
class PILDataset(Dataset):
    def __init__(self, path, img_size):
        self.path = [os.path.join(path, i) for i in os.listdir(path)]
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_path = self.path[index]
        img_name = os.path.basename(img_path)
        img_label = img_name.split('.')[0]
        label = 0 if img_label == 'cat' else 1
        label = np.array([label], dtype=np.float32)
        img = Image.open(img_path).convert("RGB")
        img, _, _ = letterbox(img, self.img_size, auto=False)
        data = self.transform(img)
        return data, label

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    test = PILDataset('E:/DataSources/DogsAndCats/train', 224)
    test.__getitem__(40)
