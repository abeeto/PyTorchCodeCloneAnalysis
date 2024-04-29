# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com

from mobilenetv3 import mobilenet_v3_large

model_path = 'pretrained_models/mobilenet_v3_large-8738ca79.pth'
large = mobilenet_v3_large(pretrained=model_path, width_mult=1.0,
                           reduced_tail=False, dilated=False)
print(large)