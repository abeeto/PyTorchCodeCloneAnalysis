import cv2
import numpy as np
import torch
from time import time


########################################## FUNCTIONS ###################################################################
def draw_rect(img, box, color=(0, 255, 0), thickness=3):
    x0, y0, x, y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return cv2.rectangle(img, (x0, y0), (x, y), color=color, thickness=thickness)


def draw_rect_pro(image, box, cls, score):
    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = colors[cls]
    class_name = classes[cls]
    score = '{:.4f}'.format(score)
    label = '-'.join([class_name, score])

    ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, box_thickness)
    cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
    cv2.putText(image, label, (xmin, ymax - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (.0, .0, .0), font_thickness)


########################################## VARIABLES ###################################################################
classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
           14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
           29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
           41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
           49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
           57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
           64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
           71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
           78: 'hair drier', 79: 'toothbrush'}
colors = np.random.uniform(0, 255, size=(80, 3))
font_scale = 0.5
font_thickness = 1
box_thickness = 2
network_input_size = 200  # the default is 640


############################################## MAIN ####################################################################
# Load Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True).fuse().eval()
model = model.autoshape()  # add autoshape wrapper IMPORTANT

# Camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # inference
    X = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = time()
    prediction = model([X], size=network_input_size)[0]
    fps = round(1 / (time() - t))

    # draw predictions
    if prediction is not None:
        for *box, conf, cls in prediction:
            draw_rect_pro(frame, box, int(cls), conf)

    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
