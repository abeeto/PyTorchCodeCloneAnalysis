import numpy as np
import cv2
import torch
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import device, classes,test_model_name,out_dir, test_image

colors = np.random.uniform(0, 255, size=(len(classes), 3))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes)) 

checkpoint = torch.load('{out_dir}/{test_model_name}.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

detection_threshold = 0.8
RESIZE_TO = (512, 512)
images = os.listdir("{}".format(test_image))



for img in images:
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(device))

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    print(len(outputs[0]['boxes']))
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        print(boxes)
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            cv2.rectangle(img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 2)
            cv2.putText(img, class_name, 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                        2, lineType=cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)