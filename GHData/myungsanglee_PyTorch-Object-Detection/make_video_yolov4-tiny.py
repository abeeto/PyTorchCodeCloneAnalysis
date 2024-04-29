import argparse
import time
import os

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.yaml_helper import get_configs
from module.yolov3_detector import YoloV3Detector
from models.detector.yolov4_tiny import YoloV4TinyV4
from utils.yolo_utils import get_tagged_img, get_target_boxes
from utils.yolov3_utils import DecodeYoloV3
from dataset.detection.yolo_dataset import YoloDataModule


def make_video(cfg, ckpt, input, output):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])
    
    if os.path.isfile(input):
        cap = cv2.VideoCapture(input)
        
        if not cap.isOpened():
            print(f'Can\'t open video file')
            return
        
    else:
        print(f'There is no input file: {input}')
        return  
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    out = cv2.VideoWriter(output, fourcc, fps, frame_size)
    
    model = YoloV4TinyV4(
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['anchors'])
    )

    if torch.cuda.is_available:
        model = model.cuda()
    
    model_module = YoloV3Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )
    model_module.eval()

    yolov3_decoder = DecodeYoloV3(cfg['num_classes'], cfg['anchors'], cfg['input_size'], conf_threshold=cfg['conf_threshold'])

    transform = A.Compose([
        A.Resize(cfg['input_size'], cfg['input_size']),
        A.Normalize(0, 1),
        ToTensorV2(),
    ])
    
    # Inference
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_x = transform(image=img)['image']
        batch_x = torch.unsqueeze(batch_x, dim=0)

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()
        
        before = time.time()
        with torch.no_grad():
            predictions = model_module(batch_x)
        boxes = yolov3_decoder(predictions)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        
        for bbox in boxes:
            class_name = 'person'
            confidence_score = bbox[4]
            cx = bbox[0] * (frame_width / cfg['input_size'])
            cy = bbox[1] * (frame_height / cfg['input_size'])
            w = bbox[2] * (frame_width / cfg['input_size'])
            h = bbox[3] * (frame_height / cfg['input_size'])
            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))
            
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
            frame = cv2.putText(frame, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20), 
                              fontFace=cv2.FONT_HERSHEY_PLAIN,
                              fontScale=1,
                              color=(0, 255, 0))
        
        out.write(frame)
        cv2.imshow('Prediction', frame)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    parser.add_argument('--input', required=True, type=str, help='input video file path')
    parser.add_argument('--output', required=True, type=str, help='output video file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    make_video(cfg, args.ckpt, args.input, args.output)
    