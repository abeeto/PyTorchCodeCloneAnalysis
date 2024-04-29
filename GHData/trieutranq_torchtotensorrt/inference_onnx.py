from tabnanny import verbose
import onnx
import onnxruntime
import numpy as np
import onnxruntime as ort
import glob
import cv2
import torch
from torchvision import transforms
from PIL import Image
#=========================================================
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
def load_data(path, shape = (64,64)):
    data = list()
    label = list()
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()]
    )
    for i in path:
        image = Image.open(i)
        image = transform(image)
        data.append(image)
        label.append(1)
    return data, label

path =  sorted(glob.glob('/home/timtran/Desktop/TensorRT/20220331_ADLT_MergeLegacy_Train/1/*'))
x_data, x_label = load_data(path, shape = (64,64))
train_data = []
for i in range(len(x_data)):
   train_data.append([x_data[i], x_label[i]])
# print(train_data)
data = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=0)

ort.set_default_logger_severity(0)
print(ort.get_device())
opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 4
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

model_onnx = '/home/timtran/Desktop/TensorRT/char_adlt.onnx'
session = onnxruntime.InferenceSession(model_onnx, sess_options=opts,providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# data = sorted(glob.glob('/home/timtran/Desktop/TensorRT/20220331_ADLT_MergeLegacy_Train/1/*'))
# transforms = transforms.Compose([
#     transforms.Resize((64,64)),
#     transforms.Grayscale(num_output_channels=3)]
# )


def evaluate_onnx(data):
  correct =0
  for i, (inputs, labels) in enumerate(data, 0):
    inputs = np.array(inputs)
    labels = np.array(labels)

    result = session.run([output_name], {input_name: inputs})
    result = np.array(result).squeeze()
    preds = np.argmax(result, axis=1)
    correct += np.sum(preds == labels.data)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(data.dataset),100. * correct / len(data.dataset)))

evaluate_onnx(data)