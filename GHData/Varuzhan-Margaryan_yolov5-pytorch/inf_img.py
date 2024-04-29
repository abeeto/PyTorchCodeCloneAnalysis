import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True).fuse().eval()
model = model.autoshape()  # add autoshape wrapper IMPORTANT

# Images
img = Image.open('img.jpg')

# Inference
prediction = model([img], size=640)[0]  # includes NMS

# Plot
img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
if prediction is not None:  # is not None
    for *box, conf, cls in prediction:  # [xy1, xy2], confidence, class
        print('class %g %.2f, ' % (cls, conf), end='')  # label
        ImageDraw.Draw(img).rectangle(box, width=3)  # plot

img.save('results.jpg')  # save


