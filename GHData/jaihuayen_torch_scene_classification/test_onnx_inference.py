import onnxruntime as ox
import torch
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torch.load('./models/model_mobilenetv2.pt', map_location = DEVICE)

# Onnx Session
onnxPath = "./models/model_mobilenetv2.onnx"
sess = ox.InferenceSession(onnxPath)
inputName = sess.get_inputs()[0].name
outputName = sess.get_outputs()[0].name

dummy_input = torch.rand((1, 3, 224, 224))
dummy_input = dummy_input.to(DEVICE)

totalNum = 100
absDiff = list()
for num in range(totalNum):

    # PyTorch and Onnx Inference

    if torch.cuda.is_available():
        torchOutputs = model(dummy_input).cpu().detach().numpy()
        onnxOutputs = sess.run(
            [outputName], {inputName: dummy_input.cpu().detach().numpy()})[0]
    else:
        torchOutputs = model(dummy_input).detach().numpy()
        onnxOutputs = sess.run(
            [outputName], {inputName: dummy_input.numpy()})[0]

    # Check Results
    absDiff.append(np.absolute(torchOutputs-onnxOutputs).mean())

print(absDiff)
