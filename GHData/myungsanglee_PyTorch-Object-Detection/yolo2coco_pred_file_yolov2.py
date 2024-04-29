import argparse
import os

import torch
from tqdm import tqdm
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.yaml_helper import get_configs
from module.yolov2_detector import YoloV2Detector
from models.detector.yolov2 import YoloV2
from utils.yolov2_utils import DecodeYoloV2
from utils.module_select import get_model


def make_pred_result_file_for_coco_map_calculator(cfg, ckpt, json_path, save_dir):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    # train_json_path = '/home/fssv2/myungsang/datasets/voc/coco_format/train.json'
    # val_json_path = '/home/fssv2/myungsang/datasets/voc/coco_format/val.json'
    
    coco = COCO(json_path)

    imgs = coco.loadImgs(coco.getImgIds())
    cats = coco.loadCats(coco.getCatIds())

    imgs_info = [[img['id'], img['file_name'], img['width'], img['height']] for img in imgs]
    cats_dict = dict([[cat['name'], cat['id']] for cat in cats])

    backbone_features_module = get_model(cfg['backbone'])(
        pretrained=cfg['backbone_pretrained'], 
        devices=cfg['devices'], 
        features_only=True, 
        out_indices=[4, 5]
    )
    
    model = YoloV2(
        backbone_features_module=backbone_features_module,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )

    if torch.cuda.is_available:
        model = model.cuda()
    
    model_module = YoloV2Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )
    model_module.eval()

    yolov2_decoder = DecodeYoloV2(cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'], conf_threshold=cfg['conf_threshold'])

    with open(cfg['names'], 'r') as f:
        class_name_list = f.read().splitlines()
    
    results = []
    results_json_path = os.path.join(save_dir, 'results.json')

    # Inference
    for (img_id, img_name, width, height) in tqdm(imgs_info):
        img_path = os.path.join('/home/fssv2/myungsang/datasets/voc/yolo_format/val', img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = img.astype(np.float32) / 255.
        batch_x = torch.FloatTensor(img).contiguous()

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()
            
        with torch.no_grad():
            predictions = model_module(batch_x)
        boxes = yolov2_decoder(predictions)
        
        for bbox in boxes:
            class_name = class_name_list[int(bbox[-1])]
            confidence_score = bbox[4]
            cx = bbox[0] * (width / cfg['input_size'])
            cy = bbox[1] * (height / cfg['input_size'])
            w = bbox[2] * (width / cfg['input_size'])
            h = bbox[3] * (height / cfg['input_size'])
            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            w = int(w)
            h = int(h)
            
            results.append({
                "image_id": img_id,
                "category_id": cats_dict[class_name],
                "bbox": [xmin, ymin, w, h],
                "score": float(confidence_score)
            })
            
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)

    img_ids = sorted(coco.getImgIds())
    cat_ids = sorted(coco.getCatIds())

    # load detection JSON file from the disk
    cocovalPrediction = coco.loadRes(results_json_path)
	# initialize the COCOeval object by passing the coco object with
	# ground truth annotations, coco object with detection results
    cocoEval = COCOeval(coco, cocovalPrediction, "bbox")
	
	# run evaluation for each image, accumulates per image results
	# display the summary metrics of the evaluation
    cocoEval.params.imgIds  = img_ids
    cocoEval.params.catIds = cat_ids
    # cocoEval.params.catIds = [1] # person id : 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    parser.add_argument('--json', required=True, type=str, help='coco format json file')
    parser.add_argument('--save_dir', required=True, type=str, help='save dir')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    make_pred_result_file_for_coco_map_calculator(cfg, args.ckpt, args.json, args.save_dir)
    