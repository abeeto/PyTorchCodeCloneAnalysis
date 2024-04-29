# import onnx
#
# # Load the ONNX model
# model = onnx.load("model.onnx")
#
# # Check that the model is well formed
# onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))
import numpy as np
import onnxruntime as ort
import torch
import torchvision

cuda = torch.cuda.is_available()
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

ort_session = ort.InferenceSession("model.onnx", providers=providers)
# meta = ort_session.get_modelmeta().custom_metadata_map  # metadata

mnist_data = torchvision.datasets.MNIST('./data/',
                                        train=False,
                                        download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))

one_data = mnist_data[0][0].unsqueeze(0).numpy()
one_data_label = mnist_data[0][1]

outputs = ort_session.run(
    None,
    {"images": one_data.astype(np.float32)},
)
print("预测的类别是否正确：", torch.Tensor(outputs[0]).max(1, keepdim=True)[1].item() == one_data_label)
print("模型的输出数值是：", outputs[0])

# 预测的类别是否正确： True
# 模型的输出数值是： [[-2.2913797e+01 -1.7695108e+01 -1.3406548e+01 -1.5655020e+01 -2.2644754e+01 -2.3998449e+01 -3.7063354e+01 -1.9073486e-06 -2.1006844e+01 -1.4032949e+01]]
