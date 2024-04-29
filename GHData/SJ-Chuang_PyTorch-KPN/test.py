from pytorchkpn.engine import DefaultPredictor
from pytorchkpn.config import get_cfg
import cv2

cfg = get_cfg()
cfg.MODEL.WEIGHTS = "./kpn-model/best_kpn.pth"
cfg.MODEL.NUM_CLASSES = 21

predictor = DefaultPredictor(cfg)

img = cv2.imread("hand_synth/train/00000001.jpg")
detections = predictor(img)

print(detections)