from model.backbones import vgg16
from model import yolov1
import torch

x = torch.randn((1,1,224,224))
vgg_16 = vgg16.VGG16(in_c=1)
print(vgg_16(x).shape)
yolo = yolov1.YOLOv1(vgg_16,7,2,3)
print(yolo(x).shape)
print(yolo)
