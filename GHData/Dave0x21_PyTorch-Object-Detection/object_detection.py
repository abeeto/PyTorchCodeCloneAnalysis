import sys

import cv2
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from PIL import Image

# Get the pretrained model from torchvision.models
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class labels from official PyTorch documentation for the pretrained model
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

CATEGORY_COLORS = [(185, 21, 71), (255, 201, 112), (41, 236, 32), (95, 143, 30), (186, 4, 137), 
    (197, 28, 143), (224, 33, 66), (252, 251, 147), (21, 142, 221), (158, 105, 156), (234, 90, 23), 
    (105, 199, 6), (20, 109, 120), (115, 185, 105), (18, 25, 146), (25, 174, 35), (54, 36, 225), 
    (22, 85, 166), (23, 221, 164), (172, 30, 15), (224, 33, 180), (58, 144, 115), (84, 121, 179), 
    (182, 235, 213), (133, 61, 171), (19, 57, 148), (125, 157, 206), (66, 115, 232), (159, 146, 3), 
    (244, 228, 34), (232, 200, 234), (144, 95, 214), (148, 85, 92), (94, 123, 156), (188, 99, 230), 
    (239, 87, 42), (138, 108, 219), (46, 124, 57), (71, 109, 157), (246, 157, 126), (61, 89, 62), 
    (206, 249, 18), (45, 220, 222), (157, 174, 236), (140, 52, 107), (40, 120, 32), (112, 28, 35), 
    (210, 148, 227), (145, 67, 170), (203, 59, 58), (101, 159, 65), (112, 68, 250), (239, 212, 156), 
    (24, 229, 152), (183, 142, 34), (7, 137, 124), (121, 226, 223), (114, 232, 181), (186, 231, 10), 
    (154, 22, 61), (18, 117, 108), (198, 20, 54), (121, 231, 154), (53, 232, 215), (172, 165, 150), 
    (31, 253, 43), (244, 138, 254), (249, 223, 15), (247, 9, 211), (173, 34, 39), (182, 178, 205), 
    (190, 255, 98), (157, 74, 129), (216, 212, 108), (50, 222, 193), (63, 5, 81), (2, 87, 181), 
    (209, 207, 231), (44, 199, 181), (182, 72, 228), (42, 175, 35), (81, 80, 89), (88, 250, 180), 
    (25, 155, 198), (224, 165, 155), (252, 188, 141), (246, 5, 124), (171, 231, 22), (1, 76, 174), 
    (131, 121, 203), (17, 70, 192)]


def get_prediction(img_path, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    prediction = model([img])
    labels = [COCO_INSTANCE_CATEGORY_NAMES[i]
              for i in list(prediction[0]['labels'].numpy())]
    boxes = [[(i[0], i[1]), (i[2], i[3])]
             for i in list(prediction[0]['boxes'].detach().numpy())]
    score = list(prediction[0]['scores'].detach().numpy())
    pred_t = [score.index(x) for x in score if x > threshold][-1]
    boxes = boxes[:pred_t+1]
    labels = labels[:pred_t+1]
    return boxes, labels, score


def object_detection(img_path, threshold=0.5, rect_th=2, text_size=1, text_th=2):
    """
    object_detection
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
          with opencv
        - the final image is displayed
    """
    boxes, labels, score = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        item_color = CATEGORY_COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(labels[i])]
        cv2.rectangle(img, boxes[i][0], boxes[i][1],
                      color=item_color, thickness=rect_th)
        txt = str(labels[i]) + ", " + str(int(score[i] * 100)) + "%"
        cv2.putText(img, txt, boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, item_color, thickness=text_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    img_path = sys.argv[-1]
    object_detection(img_path)
