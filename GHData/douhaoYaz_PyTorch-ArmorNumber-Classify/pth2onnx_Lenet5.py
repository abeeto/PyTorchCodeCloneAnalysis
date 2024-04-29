"""
Created on Sat Jul 17 20:36:14 2021

@author: 逗号@东莞理工ACE实验室
"""

import torch
import torchvision
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from Lenet5 import Lenet5

#model = torchvision.models.resnet50(pretrained=True)
model=Lenet5()
model.load_state_dict(torch.load("Lenet5_ArmorNum.pth"))

model.eval()

batch_size = 1
# example = torch.rand(1, 3, 224, 224)
example = torch.rand(batch_size, 3, 48, 48, requires_grad=True)

# print output with the purpose of comparing pth and onnx
output_pth = model(example)
print('output_pth:', output_pth)

# --------------------------------
export_onnx_file = "Lenet5_v1.onnx"
torch.onnx.export(model,
                  example,
                  export_onnx_file,
                  export_params=True,
                  # verbose=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output']
                  )

# 使用ONNX的api检查ONNX模型
# 加载保存的模型并输出onnx.ModelProto结构
onnx_model = onnx.load("Lenet5_v1.onnx")
# 验证模型的结构并确认模型具有有效的架构
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("Lenet5_v1.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name:to_numpy(example)}
ort_outputs = ort_session.run(None, ort_inputs)

print('output_onnx:', ort_outputs)

# compare ONNX Runtime and Pytorch results
np.testing.assert_allclose(to_numpy(output_pth), ort_outputs[0], rtol=1e-03, atol=1e-05)

print('Exported model has been tested with ONNXRuntime, and the result looks good!')
