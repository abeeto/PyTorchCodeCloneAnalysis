import torch
import torch.nn as nn
from torchvision.models import vgg19


class TotalLoss(nn.Module):
    def __init__(self, device, img_shape, threshold=70 / 127, vgg_layer_idx=21, c_weight=1e-7):
        """
        if vgg_layer_idx=21, it forwards to conv_layer(conv+relu)4_1
        """
        super().__init__()

        self.i_loss = InvertibilityLoss()
        self.g_loss = GrayscaleConformityLoss(device, img_shape, threshold, vgg_layer_idx, c_weight)
        self.q_loss = QuantizationLoss()

    def forward(self, gray_img, original_img, restored_img, loss_stage):
        i_loss = self.i_loss(original_img, restored_img)

        if loss_stage == 1:
            g_loss = self.g_loss(gray_img, original_img, ls_weight=0.5)
            total_loss = 3 * i_loss + g_loss
            q_loss = 0          # for debugging
        elif loss_stage == 2:
            g_loss = self.g_loss(gray_img, original_img, ls_weight=0.1)
            q_loss = self.q_loss(gray_img)
            total_loss = 3 * i_loss + g_loss + (10 * q_loss)

        # print("Total %4f | Invert %4f | Gray %4f | Quant %4f" % (float(total_loss), float(i_loss), float(g_loss), float(q_loss)))
        return total_loss


class InvertibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, original_img, restored_img):
        return self.loss(original_img, restored_img)


class GrayscaleConformityLoss(nn.Module):
    def __init__(self, device, img_shape, threshold, vgg_layer_idx, c_weight):
        super().__init__()

        self.threshold = threshold
        self.vgg = nn.DataParallel(vgg19(pretrained=True).features[:vgg_layer_idx], output_device=device["images"]).to(device["network"])
        self.mse = nn.MSELoss()

        self.c_weight = c_weight
        self.zeros = torch.zeros(img_shape).to(device["images"])

    def lightness(self, gray_img, original_luminance):
        # print(original_luminance.shape, gray_img.shape, self.zeros.shape)
        # loss = torch.mean(torch.max(torch.abs(original_luminance.repeat(1, 3, 1, 1) - gray_img) - self.threshold, self.zeros))
        loss = torch.mean(torch.max(torch.abs(original_luminance - gray_img) - self.threshold, self.zeros))
        return loss

    def contrast(self, gray_img, original_img):
        def _rescale(img):
            img = (img + 1) / 2 * 255
            img[:, 0, :, :] = img[:, 0, :, :] - 123.68      # subtract vgg mean following the implementation by authors(meaning?)
            img[:, 1, :, :] = img[:, 1, :, :] - 116.779
            img[:, 2, :, :] = img[:, 2, :, :] - 103.939
            return img
        # vgg_g = self.vgg(_rescale(gray_img))
        vgg_g = self.vgg(_rescale(gray_img.repeat(1, 3, 1, 1)))
        vgg_o = self.vgg(_rescale(original_img))
        return self.mse(vgg_g, vgg_o)

    def local_structure(self, gray_img, original_luminance):
        def _tv(img):
            h_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
            w_diff = img[:, :, :, 1:] - img[:, :, :, :-1]

            sum_axis = (1, 2, 3)
            var = torch.abs(h_diff).sum(dim=sum_axis) + torch.abs(w_diff).sum(dim=sum_axis)

            return var
        gray_tv = torch.mean(_tv(gray_img) / (256 ** 2))
        original_tv = torch.mean(_tv(original_luminance) / (256 ** 2))
        loss = abs(gray_tv - original_tv)
        return loss

    def forward(self, gray_img, original_img, ls_weight):
        r, g, b = [ch.unsqueeze(1) for ch in torch.unbind(original_img, dim=1)]
        original_luminance = (.299 * r) + (.587 * g) + (.114 * b)

        l_loss = self.lightness(gray_img, original_luminance)
        c_loss = self.contrast(gray_img, original_img)
        ls_loss = self.local_structure(gray_img, original_luminance)

        # print("\n\tlight :", float(l_loss), " | contrast :", self.c_weight, float(c_loss), " | local :", ls_weight, float(ls_loss))
        return l_loss + (self.c_weight * c_loss) + (ls_weight * ls_loss)


class QuantizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gray_img):
        min_tensor = torch.zeros_like(gray_img).fill_(1)   # fill with maximum value (larger than 255)

        for i in range(0, 256):
            min_tensor = torch.min(min_tensor, torch.abs(gray_img - (i / 127.5 - 1)))

        loss = torch.mean(min_tensor)
        return loss
