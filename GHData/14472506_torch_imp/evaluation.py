import torchvision
import torchvision.transforms as T
import torch

from models import MaskRCNN_model, MaskRCNN_mobilenetv2
import utils
from data_loader import COCOLoader

import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import sys
import time

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset



def data_loader_config(dir, batch_size):
    """
    funttion task: to configure the data loader using only one string to reduce inputs in the 
                   config dictionary. the function makes the assumption that json is titled
                   the same as the file is it located in. i.e "train".

    inputs: (dir[str]) - A string used to get the json string and to point the COCO loader at 
                         the directory where the data is stored
            
            (batch_size[int]) - passed to the data loader function
    
    outputs: returns a dataload that parses the coco dataset pointed to by the dir
    
    dependancies: - COCOLoader function from data_loader.py
    """
    
    # configuring json string
    json = "/" + dir.split("/")[-1] + ".json"
    
    # loading dataset
    dataset = COCOLoader(dir, dir + json)

    # configuring data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # returing data loader
    return(data_loader)
  
  

def get_coloured_mask(mask):
  """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)
  r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask



def get_prediction(img_path, confidence, COCO_CLASS_NAMES, model):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - confidence - threshold to keep the prediction or not
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0
    
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  # print(pred[0]['labels'].numpy().max())
  pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class



def segment_instance(img_path, COCO_CLASS_NAMES, model, confidence=0.5, rect_th=2, text_size=2, text_th=2):
  """
  segment_instance
    parameters:
      - img_path - path to input image
      - confidence- confidence to keep the prediction or not
      - rect_th - rect thickness
      - text_size
      - text_th - text thickness
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
  """
  masks, boxes, pred_cls = get_prediction(img_path, confidence, COCO_CLASS_NAMES, model)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i in range(len(masks)):
    rgb_mask = get_coloured_mask(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    b1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
    b2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
    cv2.rectangle(img, b1, b2, color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img,pred_cls[i], b1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()



@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types



def fps_performance(model, image_path, device):
    
    # loading image
    img = Image.open(image_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)

    # init times list
    times = []

    for i in range(10):
        start_time = time.time()
          
        pred = model([img])
        
        delta = time.time() - start_time
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    return(fps)



def main(model_path, num_classes, img_path, data_path, seg_instance=False, coco_eval_key=False, fps_eval=False):
  
    # This line should be ran first to ensure a gpu is being used if possible
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    # fixing the random seed: 42 is the key!
    utils.fix_seed(42)

    # load model: TODO, put this somewhere else i.e in models
    model = MaskRCNN_model(num_classes) 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # set model to eval mode
    model.eval()

    # define category names
    Labels = ['__background__', 'jersey_royal']
    
    # getting data_loader
    test_data_loader = data_loader_config(data_path, 2)
    
    if coco_eval_key:
      evaluate(model, test_data_loader, device=device) 
    
    if fps_eval:
      model.to(device)    
      fps = fps_performance(model, img_path, device)
      print(fps)

    if seg_instance:
      segment_instance(img_path,
                       Labels,
                       model,
                       confidence=0.5,
                       rect_th=2,
                       text_size=1,
                       text_th=2
                       )
    


if __name__ == "__main__":
    
    model_path = "output/Mask_RCNN_R50_test/checkpoints/best_model.pth"
    num_classes = 2
    img_path = "data/jersey_royal_dataset/test/162.JPG"
    data_path = "data/jersey_royal_dataset/test"

    main(model_path, num_classes, img_path, data_path, fps_eval=True)