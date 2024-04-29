import os
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
import onnx
import onnxruntime
print(onnx.__version__)
print(onnxruntime.__version__)


# x = torch.randn(1, 3, 512, 512)
# x = torch.randn(1, 3, 416, 416)
# model_path = '/home/fssv2/myungsang/my_projects/pytorch-YOLOv4/yolov4_1_3_512_512_static.onnx'
model_path = '/home/fssv2/myungsang/my_projects/tensorrt_demos/yolo/yolov4-tiny-3l-512.onnx'
# model_path = './yolov3_voc.onnx'
onnx_model = onnx.load(model_path)
# onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

tmp_file_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/tmp.txt'
with open(tmp_file_path, 'r') as f:
    data_list = f.read().splitlines()
print(len(data_list))

fps_list = []
for img_path in data_list:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32)

    # ONNX 런타임에서 계산된 결과값
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    
    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    fps = int(1/(time.time() - start))
    fps_list.append(fps)
    print(f'\rInference: {fps}', end='')
    # print(ort_inputs)
    # print(len(ort_outs))
print(f'\nAvg_FPS: {sum(fps_list)/len(fps_list)}')
