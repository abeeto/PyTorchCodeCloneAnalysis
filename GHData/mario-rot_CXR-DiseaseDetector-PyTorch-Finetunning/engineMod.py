import math
import sys
import time
from types import LambdaType
import torch
import os
import pathlib
import json
import pandas as pd

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

from torchmets import tm

def save_json(obj, path: pathlib.Path):
    with open(path, "w") as fp:
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


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


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, run, torch_mets = None):

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    labs = []
    targs = []

    # if not os.path.exists('Predictions/EP' + str(epoch)):
    #   os.makedirs('Predictions/EP' + str(epoch))
    if not os.path.exists('Predictions'):
        os.makedirs('Predictions')

    # if epoch == 0:
    #   PredDict = pd.DataFrame()
    # else:
    #   PredDict = pd.read_csv('/content/Predictions.csv', index_col=0)
    PredDict = pd.DataFrame()

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs2 = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        res2 = {target["x_name"].item(): output for target, output in zip(targets, outputs2)}
        r = str(list(res2.keys())[0])
        l = 11 - len(r)
        z = '0' * l
        zr = z + r
        image_name = zr[:-3] + '_' + zr[-3:]
        # save_json(res2[list(res2.keys())[0]],pathlib.Path("Predictions/EP"+str(epoch)+"/"+image_name+".json"))
        # run["Predictions/EP"+str(epoch)+"/"+image_name+".json"].upload("Predictions/EP"+str(epoch)+"/"+image_name+".json")
        res2[list(res2.keys())[0]]["Image_id"],res2[list(res2.keys())[0]]["Epoch"] = image_name,epoch
        PredDict = PredDict.append(res2[list(res2.keys())[0]], ignore_index=True)

        labs.append(res[list(res.keys())[0]]['labels'])
        ts = [di['labels'] for di in targets]
        # print('TS:', ts)
        targs.append(ts[0])

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    PredDict.to_csv('/content/Predictions/Epoch_'+str(epoch)+'_Predictions.csv')
    run["Predictions/Epoch_"+str(epoch)+"_Predictions.csv"].upload('/content/Predictions/Epoch_'+str(epoch)+'_Predictions.csv')

    labs = [torch.unique(item) for item in labs]
    pred_cls_oh = torch.zeros(len(labs),model.roi_heads.box_predictor.cls_score.out_features)
    for i in range(len(labs)):
      pred_cls_oh[i, labs[i]] = 1
    pred_cls_oh = pred_cls_oh[:,1:].int()
    # print(pred_cls_oh)

    targs = [torch.unique(item) for item in targs]
    gt_cls_oh = torch.zeros(len(targs),model.roi_heads.box_predictor.cls_score.out_features)
    for i in range(len(labs)):
      gt_cls_oh[i, targs[i]] = 1
    gt_cls_oh = gt_cls_oh[:,1:].int()
    # print(gt_cls_oh)

    if torch_mets:
        for met in torch_mets[0]:
            acc,pres,rec,f1 = tm(pred_cls_oh, gt_cls_oh, model.roi_heads.box_predictor.cls_score.out_features-1, met, mdmc = torch_mets[1], prnt = torch_mets[2])
            mets = {'Accuracy':acc,
                    'Precision':pres,
                    'Recall':rec,
                    'F1-Score':f1}
            for key, value in mets.items():
              run["logs/{}_{}_Method".format(key,met)].log(value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
