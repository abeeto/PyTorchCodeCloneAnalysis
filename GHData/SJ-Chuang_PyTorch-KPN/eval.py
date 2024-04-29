from pytorchkpn.engine import DefaultTrainer
from pytorchkpn.data import DatasetCatalog
from pytorchkpn.config import get_cfg
from pytorchkpn.evaluation import do_evaluation
import numpy as np
import cv2, os, json
import torch

class DataList:
    def __init__(self, image_path):
        self.image_path = image_path
        
    def __call__(self):
        data_list = []
        global max_id
        for file in os.listdir(self.image_path):
            name, ext = os.path.splitext(file)
            if ext == ".json":
                label = json.load(open(os.path.join(self.image_path, file)))
                h, w = cv2.imread(os.path.join(self.image_path, name+".jpg")).shape[:2]
                annos = []
                for idx, (x, y, visable) in enumerate(label["hand_pts"]):
                    if visable:
                        annos.append({
                            "keypoint": [x, y],
                            "category_id": idx
                        })
                        
                data_list.append({
                    "image": os.path.join(self.image_path, name+".jpg"),
                    "height": h,
                    "width": w,
                    "annotations": annos
                })
                
        return data_list

DatasetCatalog.register("train", DataList("hand_synth/train"))
DatasetCatalog.register("val", DataList("hand_synth/val"))

cfg = get_cfg()
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.VAL = ("val",)
cfg.SOLVER.EPOCH = 300
cfg.MODEL.NUM_CLASSES = 21

trainer = DefaultTrainer(cfg)
model = trainer.model
model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best_kpn.pth")))
results = do_evaluation(cfg, model, dataset_name="val")

print(results)