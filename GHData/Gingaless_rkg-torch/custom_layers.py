
import math
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
import stylegan1.custom_layers as stg1cl
from stylegan1.c_utils import calc_pool2d_pad
from kornia.filters import filter2D
import numpy as np

stg1cl.default_conv_weight_norm = None
stg1cl.default_fc_weight_norm = None
default_up_sample = partial(F.interpolate, scale_factor=2, mode='binear')
image_channels = 3
leaky_relu_alpha = 0.2

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        blur_kernel = torch.Tensor([[
            [0.1,0.2,0.1],
            [0.2,0.4,0.2],
            [0.1,0.2,0.1]
        ]])
        self.register_buffer('blur_kernel', blur_kernel)
    def forward(self, x):
        return filter2D(x, self.blur_kernel,normalized=True)


class EqualConv2D(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, kernel_size, *args, **kwargs):

        super().__init__()
        self._conv = stg1cl.PadConv2D(input_size, in_channels, out_channels, kernel_size if input_size > kernel_size else input_size, *args, **kwargs)
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size ** 2)

        self.stride = self._conv.stride

        nn.init.kaiming_normal_(self._conv.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self._conv(x)*self.scale


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, lr_mul = 1, activation=None, *args, **kwargs):

        super().__init__()
        self._linear = nn.Linear(in_dim, out_dim, *args, **kwargs)
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation
        nn.init.kaiming_normal_(self._linear.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self._linear.bias)

    def forward(self, x):
        out = self._linear(x)
        return out if self.activation==None else self.activation(out)

class StyleConv2D(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False):

        super().__init__()
        self.eps = 10e-6
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        #self.scale = 1.0 / math.sqrt(in_channels * kernel_size**2)
        weight = nn.Parameter(torch.randn(1,out_channels,in_channels,kernel_size,kernel_size))
        self.register_parameter('weight',weight)
        self.blur = Blur()
        self.modulation = EqualLinear(style_dim, in_channels)
        self.demodulate = demodulate
        self.upsample = stg1cl.UpSamplingBlock() if upsample else None
        self.downsample = stg1cl.PadAvgPool2d(input_size) if downsample else None
        self.padding = calc_pool2d_pad(self.input_size,kernel_size,1)
        self.upsamp_padding = calc_pool2d_pad(2*self.input_size,kernel_size,1)
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, style):

        batch, in_chan, h, w = x.size()
        style = self.modulation(style) + 1.0
        weight = self.weight.repeat(batch,1,1,1,1)
        if self.demodulate:
            style = style.view(batch,1,in_chan,1,1)
            weight = weight * style
            demod = torch.rsqrt(weight.pow(2).sum([2,3,4],keepdim=True)+self.eps)
            weight = weight * demod
        else:
            x = x*style.view(batch,in_chan,1,1)
        weight = weight.view(batch*self.out_channels,self.in_channels,self.kernel_size,self.kernel_size)
        res = x

        if self.upsample is not None:
            res = self.upsample(res)
            res = self.blur(res)
            res = res.view(1,batch*self.in_channels,2*h,2*w)
            res = F.pad(res,self.upsamp_padding)
            res = F.conv2d(res,weight,stride=1,groups=batch)
            res = res.view(batch,self.out_channels,2*h,2*w)
            return res
        elif self.downsample is not None:
            res = res.view(1, batch*self.in_channels, h, w)
            res = F.pad(res, self.padding)
            res = F.conv2d(res,weight,stride=1,groups=batch)
            res = res.view(batch,self.out_channels,h,w)
            res = self.downsample(res)
            return res
        else:
            res = res.view(1,batch*self.in_channels,h,w)
            res = F.pad(res,self.padding)
            res = F.conv2d(res,weight,stride=1,groups=batch)
            res = res.view(batch,self.out_channels,h,w)
            return res

class toRGB(StyleConv2D):

    def __init__(self, input_size, in_channels, style_dim):
        super().__init__(input_size, in_channels, image_channels, 1, style_dim, demodulate=False)


class StyleUpSkipBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, style_dim, upsample=True,self_attn=False):
        super().__init__()
        self.register_parameter('b1',nn.Parameter(torch.zeros(1,out_channels,1,1)))
        self.register_parameter('b2',nn.Parameter(torch.zeros(1,out_channels,1,1)))
        self.register_parameter('b3',nn.Parameter(torch.zeros(1,out_channels,1,1)))
        #self.register_parameter('prev_scale',nn.Parameter(torch.ones(1)))
        self.input_size = input_size
        self.output_size = input_size*2 if upsample else input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.upsample_prev = nn.Sequential(stg1cl.UpSamplingBlock(),Blur()) if upsample else None
        self.blur = Blur() if upsample else None
        #self.activation1, self.activation2, self.activation3 = [nn.LeakyReLU(leaky_relu_alpha) for _ in range(3)]
        self.activation = nn.LeakyReLU(leaky_relu_alpha)
        self.noise1, self.noise2, self.noise3 = [stg1cl.Scale_B(out_channels) for _ in range(3)]
        self.conv1 = StyleConv2D(input_size, in_channels, out_channels, 3, style_dim)
        self.conv2 = StyleConv2D(input_size, out_channels, out_channels, 3, style_dim, upsample=upsample)
        #self.conv2 = StyleConv2D(self.output_size, out_channels, out_channels, 3, style_dim)
        self.conv3 = StyleConv2D(self.output_size, out_channels, out_channels, 3, style_dim)
        self.to_rgb = toRGB(self.output_size, out_channels, style_dim)
        self.self_attn = stg1cl.SelfAttention(out_channels) if self_attn else None
        self.register_parameter('t',nn.Parameter(torch.zeros(1)))
    '''
    def _apply(self, fn):
        super(StyleUpSkipBlock, self)._apply(fn)
        self.b1, self.b2, self.b3 = fn(self.b1), fn(self.b2), fn(self.b3)
        return self
    '''

    def unit_operation(self, conv_layer, bias, noise_layer, activation, input_feature_map, style, noise):
        out_ft_map = conv_layer(input_feature_map, style)
        '''
        if conv_layer.upsample is not None:
           out_ft_map = self.blur(out_ft_map)
        '''
        #out_ft_map = out_ft_map + bias
        if noise is not None:
            out_ft_map = out_ft_map + noise_layer(noise)
        if activation is None:
            out_ft_map = out_ft_map + bias
        else:
            out_ft_map = activation(out_ft_map + bias)
        return out_ft_map

    def forward(self, input_feature_map, style, prev_rgb=None, noise=None, res_log2=1.0):
        #out_ft_map = self.unit_operation(self.conv1, 0.0, self.noise1, None, input_feature_map.clone(), style, noise) # output feature map
        out_ft_map = self.unit_operation(self.conv1, self.b1, self.noise1, self.activation, input_feature_map.clone(), style, noise)
        if self.self_attn is not None:
            out_ft_map = self.self_attn(out_ft_map)
        out_ft_map = self.unit_operation(self.conv2, self.b2, self.noise2, self.activation, out_ft_map, style, noise)
        out_ft_map = self.unit_operation(self.conv3, self.b3, self.noise3, self.activation, out_ft_map, style, noise)
        out_rgb = self.to_rgb(out_ft_map, style)
        if prev_rgb is not None:
            if self.upsample_prev is not None:
                prev_rgb = self.upsample_prev(prev_rgb)
            '''
            res = torch.log2(torch.Tensor(np.array([out_rgb.size(2)])))
            lod = res_log2 - res
            lod_in = 3
            '''
            out_rgb = out_rgb + prev_rgb
            #out_rgb = (1-self.t)*out_rgb + self.t*prev_rgb
            #out_rgb = out_rgb + (prev_rgb - out_rgb)*torch.clamp(lod_in - lod,0.0,1.0)
        return out_ft_map, out_rgb if out_ft_map.size(1) > image_channels else out_rgb

class ResDownBlock(stg1cl.ResDownBlock):
    def __init__(self, input_size, in_channel, out_channel, pooling='avg', alpha=0.2, self_attn=False):
        super().__init__(input_size, in_channel, out_channel, pooling, alpha)
        #modules = [EqualConv2D(input_size, in_channel, out_channel, 3),
        #nn.LeakyReLU(alpha), EqualConv2D(input_size, out_channel, out_channel, 3), nn.LeakyReLU(alpha),EqualConv2D(input_size, out_channel, out_channel, 1)]
        modules = [stg1cl.PadConv2D(input_size, in_channel, out_channel, 3), nn.LeakyReLU(alpha), stg1cl.PadConv2D(input_size, out_channel, out_channel,3), nn.LeakyReLU(alpha)]
        if self_attn:
            modules = [stg1cl.SelfAttention(in_channel)] + modules
        self.self_attn = self_attn
        self.func = nn.Sequential(*modules)
        self.mod_id = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return super().forward(x)*torch.rsqrt(torch.Tensor([2.0]).to(x.device))

class NormalizeW(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return F.normalize(x,dim=1)

class IntermediateG(stg1cl.IntermediateG):
    def __init__(self,n_fc, dim_latent):
        super().__init__(n_fc,dim_latent,leaky_relu_alpha)
        layers=[NormalizeW()]
        for _ in range(n_fc):
            layers.append(EqualLinear(dim_latent,dim_latent))
            layers.append(nn.LeakyReLU(leaky_relu_alpha))
        self.mapping = nn.Sequential(*layers)


if __name__=='__main__':
    
    import stylegan1.c_dset as dset
    
    img_path = 'p/'
    sty = torch.randn(4,32)
    styupskip1 = StyleUpSkipBlock(128,3,16,32)
    styupskip2 = StyleUpSkipBlock(256,16,8,32,upsample=False)
    act = nn.LeakyReLU(0.2)
    to_rgb = toRGB(256,8,32)
    img_tensor = next(iter(dset.create_image_loader_from_path(img_path, 128, 4)))[0]
    obj_tensor = F.interpolate(img_tensor,scale_factor=2.0)
    params = list(styupskip1.parameters()) + list(styupskip2.parameters()) + list(to_rgb.parameters())
    opt = torch.optim.Adam(params,lr=0.01)
    for param in styupskip1.parameters():
        print(param.name, ':', param.size())
    #exit()
    dset.show_images_from_tensors(obj_tensor,(2,2))
    t_epochs = 100
    show_loss = 10
    for epoch in range(t_epochs):
        styupskip1.zero_grad()
        styupskip2.zero_grad()
        to_rgb.zero_grad()
        prev_rgb = None
        res,prev_rgb = styupskip1(img_tensor,sty)
        res = act(res)
        res,prev_rgb = styupskip2(res,sty,prev_rgb=prev_rgb)
        res = act(res)
        res = to_rgb(res,sty) + prev_rgb
        ls_loss = torch.mean((res-obj_tensor).pow(2))
        ls_loss.backward()
        opt.step()
        if epoch % show_loss == 0 or epoch==t_epochs-1:
            dset.show_images_from_tensors(res.detach(),(2,2))
            print(ls_loss.item())
        