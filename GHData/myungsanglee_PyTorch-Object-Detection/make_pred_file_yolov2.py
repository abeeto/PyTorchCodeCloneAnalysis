import argparse
import os

import torch
from tqdm import tqdm

from utils.yaml_helper import get_configs
from module.yolov2_detector import YoloV2Detector
from models.detector.yolov2 import YoloV2
from utils.yolov2_utils import DecodeYoloV2
from dataset.detection.yolo_dataset import YoloDataModule
from utils.module_select import get_model


def make_pred_result_file_for_public_map_calculator(cfg, ckpt, save_dir):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = YoloDataModule(
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
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]

    img_num = 0
    # Inference
    for sample in tqdm(data_module.val_dataloader()):
        batch_x = sample['img']

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()
        
        with torch.no_grad():
            predictions = model_module(batch_x)
        boxes = yolov2_decoder(predictions)
        
        img_num += 1
        pred_txt_fd = open(os.path.join(save_dir, f'{img_num:05d}.txt'), 'w')
        
        for bbox in boxes:
            class_name = class_name_list[int(bbox[-1])]
            confidence_score = bbox[4]
            cx = bbox[0]
            cy = bbox[1]
            w = bbox[2]
            h = bbox[3]
            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))

            pred_txt_fd.write(f'{class_name} {confidence_score} {xmin} {ymin} {xmax} {ymax}\n')
        pred_txt_fd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    parser.add_argument('--save_dir', required=True, type=str, help='save dir')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    make_pred_result_file_for_public_map_calculator(cfg, args.ckpt, args.save_dir)
