import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_transform as tr

import kornia

# Many losses in this model require passing through VGG19, define once in init to pass on, rather than creating many classes
# to load more than once into memory


class Loss(nn.Module):
    """Wraps all loss function into one class."""

    def __init__(self, resize=False, normalize_mean_std=True, device='cpu'):
        super(Loss, self).__init__()

        self.model = models.vgg19(pretrained=True).features[:26].to(
            device).eval()  # features[:26] get to conv4_4
        for i in range(26):
            layer = self.model[i]
            if isinstance(layer, nn.MaxPool2d):
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2)
                # Convert MaxPool to AvgPool
        for p in self.model.parameters():
            p.requires_grad = False

        self.blocks = nn.ModuleList([self.model[0],
                                     self.model[2],
                                     self.model[4:6]])

        self.normalize_mean_std = normalize_mean_std
        if self.normalize_mean_std:
            self.mean = nn.Parameter(torch.tensor(
                [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
            self.std = nn.Parameter(torch.tensor(
                [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

        self.resize = resize

    def gram_matrix(self, input):
        """Compute Gram Matrix."""
        a, b, c, d = input.size()

        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def content_loss(self, input, target, per: bool = True):
        """Real image & generated image."""
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        if self.normalize_mean_std:
            input = torch.clip(tr.inv_norm(input), 0, 1)  # [-1,1]->[0,1]
            target = torch.clip(tr.inv_norm(target), 0, 1)
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std

        if not per:
            input = self.model(input)
            target = self.model(target).detach()
        else:
            input = self.model[:16](input)
            target = self.model[:16](target).detach()
            '''Use Original perceptual loss relu3_3'''

        return F.l1_loss(input, target)

    def style_loss(self, input, target, grayscale=False, luma=True):
        '''Style image & generated image.'''

        if luma:
            input = torch.clip(tr.inv_norm(input), 0, 1)
            target = torch.clip(tr.inv_norm(target), 0, 1)
            input = kornia.rgb_to_ycbcr(input)[:, 0].unsqueeze(1)
            target = kornia.rgb_to_ycbcr(target)[:, 0].unsqueeze(1)

        elif grayscale:
            input = tr.inv_gray_transform(input)
            target = tr.inv_gray_transform(input)

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        if self.normalize_mean_std:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        input = self.model(input)
        input_gram = self.gram_matrix(input)
        target = self.model(target).detach()
        target_gram = self.gram_matrix(target)

        return F.l1_loss(input_gram, target_gram)

    def perceptual_loss(self, input, target, init_weight=1/32.):
        r'''ESTHER
            real image & transformed image'''

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        if self.normalize_mean_std:
            input = torch.clip(tr.inv_norm(input), 0, 1)
            target = torch.clip(tr.inv_norm(target), 0, 1)
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        weight = init_weight

        total_loss = 0

        for block in self.blocks:
            input = block(input)
            target = block(target).detach()
            loss = F.l1_loss(input, target) * weight
            weight *= 2.
            # we manually pass convolved image thorugh relu, instead of passing through blocks
            input = F.relu(input)
            target = F.relu(target)  # this is because it is set to inplace, hence will cause error
            total_loss += loss

        return total_loss

    def reconstruction_loss(self, input, target):
        input = torch.clip(tr.inv_norm(input), 0, 1)
        target = torch.clip(tr.inv_norm(target), 0, 1)
        # kornia function requires in range [0-1]
        input = kornia.rgb_to_ycbcr(input)
        target = kornia.rgb_to_ycbcr(target).detach()
        loss = F.l1_loss(input[:, 0], target[:, 0])
        loss += F.smooth_l1_loss(input[:, 1:], target[:, 1:])

        return loss

    def tv_loss(self, input):
        """Total Variation Loss."""
        tv_loss = torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
            torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return tv_loss
