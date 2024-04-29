# content loss difination
# target 是目标图片等内容表征
# input 是输入噪声的内容表征
# 计算两个表征的MSE
# backward 更新input


# 输入值规范到0-1之间，提高收敛率


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 如果没有gpu，那就用cpu
imsize = 512 if torch.cuda.is_available() else (200, 200)

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0) #add a fake batch dimension
    return image.to(device, torch.float)


style_img = image_loader("./picasso.jpg")
content_img = image_loader("./IMG_4573.jpeg")


# show the resized image
unloader = transforms.ToPILImage() # reconvert into PIL image
plt.ion() # 打开交互模式


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    
plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

# all module that will be in between layers must have a forward method

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
 
    


# define gram matrix
# G = A.t()*A
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# style loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        # the gram_matrix of target_feature

    def forward(self, input):
        G = gram_matrix(input) # the gram_matrix of input
        self.loss = F.mse_loss(G, self.target)
        return input
 
    


#biuld network
'''PyTorch’s implementation of VGG is a module divided into two child 
Sequential modules: features (containing convolution and pooling layers), 
and classifier (containing fully connected layers). We will use the features 
module because we need the output of the individual convolution layers to measure 
content and style loss. Some layers have different behavior during training than evaluation, 
so we must set the network to evaluation mode using .eval().
'''
cnn = models.vgg19(pretrained=True).features.to(device).eval() #模型评价
'''Additionally, VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] 
and std=[0.229, 0.224, 0.225]. We will use them to normalize the image before sending it into the network.
'''
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W]. where B = 1
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std



# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization) 
    # 把normalization层添加的model的开始，参数一传进来就先normalization

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i) #如果第i层是nn.Conv2d类，则name = 'conv_i'
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer) # 慢慢的把每一层model加进去

        if name in content_layers:
            # 如果属于content_layer
            # add content loss:
            target = model(content_img).detach() #计算content_img 在这一层的内容表征
            content_loss = ContentLoss(target) # 先把target传入ContentLoss, 形成一个object
            model.add_module("content_loss_{}".format(i), content_loss) 
            # 把content_loss类以“conten_loss_i“的名字 加在当前layer后面
            content_losses.append(content_loss) #并把这个类放在 content_losses列表中

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off（修剪）the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        # 范围从 len(model)到 0 (不含-1)， 步长为-1
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break # i 停留在最后一个, 是ContentLoss或者StyleLoss的层上

    model = model[:(i + 1)] #提取model中从最开始到第i个

    return model, style_losses, content_losses


input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)
# content_img是一个variable, .data使它变成tensor, .size获得它的形状


# add the original input image to the figure:
#plt.figure()
#imshow(input_img, title='Input Image')

'''
As Leon Gatys, the author of the algorithm, suggested here, we will use L-BFGS algorithm to run our gradient descent. 
Unlike training a network, we want to train the input image in order to minimise the content/style losses. 
We will create a PyTorch L-BFGS optimizer optim.LBFGS and pass our image to it as the tensor to optimize.
'''

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# The optimizer requires a “closure” function, which reevaluates the modul and returns the loss.

'''
We still have one final constraint to address. 
The network may try to optimize the input with values that exceed the 0 to 1 tensor range for the image. 
We can address this by correcting the input values to be between 0 to 1 each time the network is run.
'''
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            # and return the loss
            input_img.data.clamp_(0, 1)
            # all bigger than 1 ->1, less than 0  -> 0
            optimizer.zero_grad()
            #梯度置0
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    # 最后一次更新之后，还是要clamp一下
    input_img.data.clamp_(0, 1)

    return input_img



output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, num_steps = 300, style_weight=700000, content_weight=1)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

























