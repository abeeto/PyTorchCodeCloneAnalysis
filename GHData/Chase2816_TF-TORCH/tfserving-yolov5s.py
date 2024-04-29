import requests
import numpy as np
import cv2
import sys, os, shutil
import json
from PIL import Image, ImageDraw, ImageFont


def nms(dets, iou_thred, cfd_thred):
    if len(dets) == 0: return []
    bboxes = np.array(dets)
    ## 对整个bboxes排序
    bboxes = bboxes[np.argsort(bboxes[:, 4])]
    pick_bboxes = []
    #     print(bboxes)
    while bboxes.shape[0] and bboxes[-1, 4] >= cfd_thred:
        # while bboxes.shape[0] and bboxes[-1, -1] >= cfd_thred:
        bbox = bboxes[-1]
        x1 = np.maximum(bbox[0], bboxes[:-1, 0])
        y1 = np.maximum(bbox[1], bboxes[:-1, 1])
        x2 = np.minimum(bbox[2], bboxes[:-1, 2])
        y2 = np.minimum(bbox[3], bboxes[:-1, 3])
        inters = np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0)
        unions = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) + (bboxes[:-1, 2] - bboxes[:-1, 0] + 1) * (
                bboxes[:-1, 3] - bboxes[:-1, 1] + 1) - inters
        ious = inters / unions
        keep_indices = np.where(ious < iou_thred)
        bboxes = bboxes[keep_indices]  ## indices一定不包括自己
        pick_bboxes.append(bbox)
    return np.asarray(pick_bboxes)


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
         'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors',
         'teddy bear', 'hair drier', 'toothbrush']

input_imgs = r"D:\CC\workplace\yolov5\data\images/"
files = os.listdir(input_imgs)
# save_path = r"E:\pycharm_project\tfservingconvert\water_flowers\outputs\out_imgs2"
save_path = r"D:\CC\workplace\yolov5\data\tfserver_out"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
for file in files:
    # input_img = r"E:\data\diode-opt\imgs\20200611_84.jpg"
    # input_img = r"E:\pycharm_project\tfservingconvert\tf1.15v3\yc960xc1484.jpg"
    if file.split('.')[1] == "xml": continue
    input_img = input_imgs + file
    img = Image.open(input_img)
    img1 = img.resize((640, 640))
    # img1 = img.resize((224, 224))
    image_np = np.array(img1)
    image_np = image_np / 255.
    img_data = image_np[np.newaxis, :].tolist()
    data = {"instances": img_data}
    preds = requests.post("http://172.20.112.102:9911/v1/models/yolov5:predict", json=data)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
    pred = np.array(predictions)

    keep = nms(pred, 0.7, 0.25)
    print(keep)
    print(keep.shape)
    cv_im = cv2.imread(input_img)
    for i, det in enumerate(keep):
        score = det[4]
        boxes = det[:4]
        conf = det[5:]
        label = det[5:].argmax()

        # if score > 0.25:
        left_x = det[0] - det[2] / 2
        left_y = det[1] - det[3] / 2
        right_x = det[0] + det[2] / 2
        right_y = det[1] + det[3] / 2

        # darw.rectangle((left_x * img1.width, left_y * img1.height, right_x * img1.width, right_y * img1.height))
        # img1.show()

        cv2.rectangle(cv_im, (int(left_x * cv_im.shape[1]), int(left_y * cv_im.shape[0])),
                      (int(right_x * cv_im.shape[1]), int(right_y * cv_im.shape[0])), (0, 255, 0))
        cv2.putText(cv_im, names[label] + "-" + str(score),
                    (int(left_x * cv_im.shape[1]), int(left_y * cv_im.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 0))
        cv2.imshow("cv_im", cv_im)
        cv2.waitKey(0)

    # exit()
    # # darw = ImageDraw.Draw(img1)
    # cv_im = cv2.imread(input_img)
    # print(cv_im.shape)
    # for i, det in enumerate(pred):
    #     print(det)
    #     score = det[4]
    #     boxes = det[:4]
    #     conf = det[5:]
    #     label = det[5:].argmax()
    #
    #     if score > 0.25:
    #
    #         left_x = det[0] - det[2] / 2
    #         left_y = det[1] - det[3] / 2
    #         right_x = det[0] + det[2] / 2
    #         right_y = det[1] + det[3] / 2
    #
    #         # darw.rectangle((left_x * img1.width, left_y * img1.height, right_x * img1.width, right_y * img1.height))
    #         # img1.show()
    #
    #         cv2.rectangle(cv_im,(int(left_x*cv_im.shape[1]),int(left_y*cv_im.shape[0])),(int(right_x*cv_im.shape[1]),int(right_y*cv_im.shape[0])),(0,255,0))
    #         cv2.putText(cv_im, names[label] + "-" + str(score), (int(left_x*cv_im.shape[1]),int(left_y*cv_im.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0))
    #         cv2.imshow("cv_im",cv_im)
    #         cv2.waitKey(0)
    #
