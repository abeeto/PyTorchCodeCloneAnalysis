from torch_snippets import *
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 3


def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=NUM_CLASSES
    )
    return model
