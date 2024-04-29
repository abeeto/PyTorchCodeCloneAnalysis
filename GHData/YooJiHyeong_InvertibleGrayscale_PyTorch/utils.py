import torch
import torchvision
from tensorboardX import SummaryWriter


class TensorboardLogger(SummaryWriter):
    def __init__(self, log_path):
        super().__init__(log_path)
        self.log_path = log_path

    def log_scalar(self, tag, scalar, global_step):
        # single scalar
        if isinstance(scalar, (int, float)):
            self.add_scalar(tag, scalar, global_step)
        # scalar group
        elif isinstance(scalar, dict):
            self.add_scalars(tag, scalar, global_step)

    def log_image(self, original_img, gray_img, restored_img, epoch, plot_name="Plot"):
        def _normalize(img):
            return (img + 1) / 2
        # imgs = [_normalize(img) for img in [original_img, gray_img, restored_img]]
        imgs = [_normalize(img) for img in [original_img, gray_img.repeat(1, 3, 1, 1), restored_img]]
        imgs = torch.cat(imgs, 3)
        imgs = torchvision.utils.make_grid(imgs, nrow=1)
        self.add_image("orig / gray / restored", imgs, epoch)
