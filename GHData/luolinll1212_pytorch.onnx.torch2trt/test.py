# -*- coding: utf-8 -*-
import torchvision
import torch
from torch2trt import torch2trt

data = torch.randn((1, 3, 224, 224)).cuda().half()
model = torchvision.models.resnet18(pretrained=True).cuda().half().eval()
output = model(data)

# pytorch -> tensorrt
model_trt = torch2trt(model, [data], fp16_mode=True)
output_trt = model_trt(data)

# compare
print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))
print("mse :%f" % float((output - output_trt)**2))

# save tensorrt model
torch.save(model_trt.state_dict(), "resnet18_trt.pth")

# load tensorrt model
from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('resnet18_trt.pth'))



