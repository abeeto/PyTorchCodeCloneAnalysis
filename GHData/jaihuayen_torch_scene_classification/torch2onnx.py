import numpy as np
import onnxruntime as ox
import torch

import torch.onnx 

model = torch.load('./models/model_mobilenetv2.pt')

# set the model to inference mode 
model.eval()

# Let's create a dummy input tensor  
dummy_input = torch.rand((1, 3, 224, 224))

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(dev)
dummy_input = dummy_input.to(dev)

input_names = ["actual_input"]
output_names = ["output"]

torch.onnx.export(model, dummy_input, "./models/model_mobilenetv2.onnx", 
                  verbose=False, input_names=input_names, 
                  output_names=output_names, export_params=True)