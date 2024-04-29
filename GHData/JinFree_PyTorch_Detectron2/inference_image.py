import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import time


def frameProcessing(im, predictor, Visualizer, cfg):
    tick0 = time.time()
    outputs = predictor(im)
    tick1 = time.time()

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv_bgr = out.get_image()[:, :, ::-1]
    tick2 = time.time()
    print(tick1 - tick0, tick2 - tick1)
    return cv_bgr

def main(imagePath):
        
    im = cv2.imread(imagePath)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    cv_bgr = None
    for i in range(5):
        
        cv_bgr = frameProcessing(im, predictor, Visualizer, cfg)

    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    cv2.imshow('input' ,im)
    cv2.imshow('output', cv_bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
if __name__=='__main__':

    imagePath = './dog.jpg'
    main(imagePath)
