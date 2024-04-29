import argparse
import time
import os

import torch
import numpy as np
import cv2

from utils.yaml_helper import get_configs
from module.yolov3_detector import YoloV3Detector
from models.detector.yolov3 import YoloV3
from utils.yolo_utils import get_tagged_img, get_target_boxes
from utils.yolov3_utils import DecodeYoloV3
from dataset.detection.yolo_dataset import YoloDataModule
from dataset.detection.datasets import DataModule
from utils.module_select import get_model


def inference(cfg, ckpt):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=1
    )
    data_module.prepare_data()
    data_module.setup()

    backbone_features_module = get_model(cfg['backbone'])(
        pretrained=cfg['backbone_pretrained'], 
        devices=cfg['devices'],
        features_only=True,
        out_indices=[3, 4, 5]
    )
    
    model = YoloV3(
        backbone_features_module=backbone_features_module,
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

    # Inference
    tmp_num = 0
    for sample in data_module.val_dataloader():
        batch_x = sample['img']
        batch_y = sample['annot']

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()
        
        before = time.time()
        with torch.no_grad():
            predictions = model_module(batch_x)
        boxes = yolov3_decoder(predictions)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        
        # batch_x to img
        if torch.cuda.is_available:
            img = batch_x.cpu()[0].numpy()   
        else:
            img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        true_boxes = get_target_boxes(batch_y, cfg['input_size'])
        
        pred_img = get_tagged_img(img.copy(), boxes, cfg['names'], (0, 255, 0))
        true_img = get_tagged_img(img.copy(), true_boxes, cfg['names'], (0, 0, 255))
        
        cv2.imshow('GT', true_img)
        cv2.imshow('Prediction', pred_img)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('c'):
            save_img_dir = f'{os.sep}'.join(ckpt.split(os.sep)[:-2])
            tmp_num += 1
            img_path = os.path.join(save_img_dir, f'captured_image_{tmp_num:02d}.jpg')
            cv2.imwrite(img_path, pred_img)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    inference(cfg, args.ckpt)
    