# -*- coding: utf-8 -*-
try:
    from efficientnet_pytorch import EfficientNet
except:
    print('Please install package efficientnet_pytorch first')


model_E0 = EfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=2) 
model_E1 = EfficientNet.from_pretrained(model_name='efficientnet-b1', num_classes=2) 
model_E2 = EfficientNet.from_pretrained(model_name='efficientnet-b2', num_classes=2) 
model_E3 = EfficientNet.from_pretrained(model_name='efficientnet-b3', num_classes=2) 
model_E4 = EfficientNet.from_pretrained(model_name='efficientnet-b4', num_classes=2) 
model_E5 = EfficientNet.from_pretrained(model_name='efficientnet-b5', num_classes=2) 
model_E6 = EfficientNet.from_pretrained(model_name='efficientnet-b6', num_classes=2) 
model_E7 = EfficientNet.from_pretrained(model_name='efficientnet-b7', num_classes=2) 

Efficient_models = [model_E0,model_E1,model_E2,model_E3,model_E4,model_E5,model_E6,model_E7]

def count_parameters(model):
    x1, x2 = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(x1, x2)
    return [x1, x2]

for i in range(8):
    print('\n')
    input_size = EfficientNet.get_image_size('efficientnet-b'+str(i))
    print('EfficientNet b'+str(i), ' has input image size: ', input_size)
    print('Model parameters: ', count_parameters(Efficient_models[i])[0])
    
#EfficientNet b0  has input image size:  224
#4010110 4010110
#Model parameters:  4010110
#
#
#EfficientNet b1  has input image size:  240
#6515746 6515746
#Model parameters:  6515746
#
#
#EfficientNet b2  has input image size:  260
#7703812 7703812
#Model parameters:  7703812
#
#
#EfficientNet b3  has input image size:  300
#10699306 10699306
#Model parameters:  10699306
#
#
#EfficientNet b4  has input image size:  380
#17552202 17552202
#Model parameters:  17552202
#
#
#EfficientNet b5  has input image size:  456
#28344882 28344882
#Model parameters:  28344882
#
#
#EfficientNet b6  has input image size:  528
#40740314 40740314
#Model parameters:  40740314
#
#
#EfficientNet b7  has input image size:  600
#63792082 63792082
#Model parameters:  63792082


from torchvision import models

net = models.MobileNetV2()
count_parameters(net)

net = models.densenet201()
count_parameters(net)

net = models.inception_v3()
count_parameters(net)
#[27161264, 27161264]

net = models.googlenet()
count_parameters(net)
#[13004888, 13004888]

net = models.vgg19_bn()
count_parameters(net)
# 143678248

net = models.vgg19()
count_parameters(net)
# 143667240


