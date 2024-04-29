import os
import glob
import colorsys
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from tqdm import tqdm
import torch.multiprocessing as mp

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', \
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', \
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', \
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', \
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', \
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', \
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', \
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', \
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', \
        'hair drier', 'toothbrush']

def applyMask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def randomColors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(11011)
    random.shuffle(colors)
    return colors

def maskInstances(image, masks, classes, all_colors):
    """Mask all instances onto the image"""
    for i in range(len(classes)):
        class_ind = classes[i]
        mask = masks[i]
        color = all_colors[class_ind]
        applyMask(image, mask, color)
    return image

def main(input_dir, input_prefix, sequences):
    """ MAIN FUNCTION """
    save_dir = "./output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        for _ in glob.glob(os.path.join(save_dir, "*.jpg")):
            os.remove(_)

    save_prefix = "detectron_"
    all_colors = randomColors(len(ALL_CLASSES))

    # model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    model_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    predictor = DefaultPredictor(cfg)

    for frame_i in tqdm(range(sequences[0], sequences[1])):
        img = cv2.imread(os.path.join(input_dir, input_prefix + "%.d.jpg"%(frame_i)))
        if img is None:
            continue
        
        outputs = predictor(img)

        # We can use `Visualizer` to draw the predictions on the image.
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # all_classes = v.metadata.thing_classes

        classes = outputs["instances"].pred_classes.cpu().numpy()
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()
        masks = pred_masks.astype(np.float)
        if len(classes) == 0:
            continue
        masked_image = maskInstances(img[..., ::-1], masks, classes, all_colors)
        cv2.imwrite(os.path.join(save_dir, save_prefix + "%.d.jpg"%(frame_i)), \
                    masked_image[:, :, ::-1])

if __name__ == "__main__":
    input_dir = "./stereo_input"
    input_prefix = "stereo_input_"
    sequences = [0, 750]
    main(input_dir, input_prefix, sequences)
