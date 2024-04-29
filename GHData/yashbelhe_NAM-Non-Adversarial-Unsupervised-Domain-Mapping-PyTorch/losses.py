import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

vgg_layers = \
['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2']

loss_layers = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

class VGG19(nn.Module):
        
    def __init__ (self, path):
        
        super(VGG19, self).__init__()

        path = './models/' + path
        
        self.bgr_mean = Variable(torch.Tensor([103.939, 116.779, 123.68]).cuda()).view(1, 3, 1, 1)
        
        self.weights = []
        self.biases = []
        
        import scipy.io
        vgg_data = scipy.io.loadmat(path)['layers'][0]
        
        for i, vgg_layer in enumerate(vgg_layers):
            
            if 'conv' in vgg_layer:

                weight = vgg_data[i][0][0][2][0][0]
                bias   = vgg_data[i][0][0][2][0][1]
                                
                weight = weight.transpose(3, 2, 0, 1)

                weight = Variable(torch.from_numpy(weight).cuda(), requires_grad=False)
                bias = Variable(torch.from_numpy(bias).cuda(), requires_grad=False)
                bias = bias.view(-1)

                self.weights.append(weight)
                self.biases.append(bias)
                
            else:
                
                self.weights.append(None)
                self.biases.append(None)

    def forward(self, x):
        
        x = x - self.bgr_mean
        
        out_layers = []

        for i, vgg_layer in enumerate(vgg_layers):
            
            if 'conv' in vgg_layer:
                
                w = self.weights[i]
                b = self.biases[i]
                x = torch.nn.functional.conv2d(x, weight=w, bias=b, stride=(1, 1), padding=(w.size()[2]//2, w.size()[3]//2))

            if 'relu' in vgg_layer:
                
                x = torch.nn.functional.relu(x)
                
            if 'pool' in vgg_layer:
                
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            
            if vgg_layer in loss_layers:
                out_layers.append(x)
            
        return out_layers

#-----------------------------------------------------------------------------------------------
#-------------------------------- perceptual diverse loss --------------------------------------
#-----------------------------------------------------------------------------------------------

def perceptual_loss(pred_layers, true_layers):
    L = len(true_layers) # number of vgg19 loss layers
    
    batch_size = pred_layers[0].size(0)

    norms_gen_img = [] # L1 norms for vgg19 layers
    for l in range(L):
        norm_gen_img = (pred_layers[l] - true_layers[l]).abs().view(batch_size, -1).mean(1)
        norms_gen_img.append(norm_gen_img)
    

    loss = torch.zeros(batch_size).cuda()
    for l in range(L):
        loss += norms_gen_img[l]

    return loss

def l1_loss(y_pred, y_true):
    batch_size = y_pred.size()[0]
    return torch.abs(y_pred - y_true).view(batch_size, -1).mean(1)