import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        
        self.real_label = target_real_label
        self.fake_label = target_fake_label

        self.real_label_tensor = None
        self.fake_label_tensor = None
        
        self.Tensor = tensor
        self.opt = opt
        
    def get_target_tensor(self, input, target_is_real):
        # print(input)
        if target_is_real: # target_is_real이 True이면 (target은 real이다.)
            if self.real_label_tensor is None: # self.real_label_tensor가 없을 때
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label) 
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input) # self.real_label_tensor 는 input 크기만큼 1.0으로 채운 tensor
            # return torch.ones_like(input)
        else: # target은 false이다.
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input) # self.fake_label_tensor 는 input 크기 만큼 0.0으로 채운 tensor
            # return torch.zeros_like(input)

    def loss(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        loss = F.binary_cross_entropy_with_logits(input, target_tensor)
        return loss
            
    def __call__(self, input, target_is_real):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        return self.loss(input, target_is_real)

# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

