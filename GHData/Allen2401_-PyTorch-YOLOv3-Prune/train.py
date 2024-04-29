from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from DarknetModel import *
import torch.optim as optim
def gather_bn_weights(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn_weights[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    return total, bn_weights


# additional subgradient descent on the sparsity-induced penalty term
def updateBN(sr):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(sr*torch.sign(m.weight.data))  # L1


def saveCheckPoint(name,state,filename):
    if "yolo" in name:
        torch.save(state, os.path.join("checkpoints/darknetgood2",filename))
    elif "res" in name:
        torch.save(state, os.path.join("checkpoints/resnet2",filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_prune_0.5_.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/autodrive.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,default="checkpoints/darknetgood2/checkpoint_6.pth",help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument('--sparsity-regularization', '-sr', default = 0.01,dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.001, help='scale sparse rate')
    parser.add_argument('--learning rate','-lr',type = float,default=1e-3, help='initial learning rate')
    opt = parser.parse_args()
    print(opt)
    logger = Logger("logs/darknetgood2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    opt.start_epoch = 0
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            checkpoint = torch.load(opt.pretrained_weights)
            if checkpoint.__len__()!=1:
                opt.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
    else:
         model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloaderss
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=0.00015)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1- epoch/80) *0.9)
   # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    metrics = [
        "grid_size","loss","x","y","w","h",
        "conf","cls","cls_acc",
        "conf50","iou50","iou75",
        "recall50","recall75","precision",
        "conf_obj","conf_noobj",
    ]

    for epoch in range(opt.start_epoch,opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()
            if opt.sr:
                updateBN(opt.sr)

            optimizer.step()
            optimizer.zero_grad()
            # ----------------
            #   Log progress
            # ----------------
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            log_str += f"\nlr {optimizer.param_groups[0]['lr']}"


            # Tensorboard loggin
            tensorboard_log = []
            for i, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    # 选择部分指标写入tensorboard
                    if name  in {"loss"}:
                        tensorboard_log += [(f"{name}_{i+1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            tensorboard_log += [("lr", optimizer.param_groups[0]['lr'])]
            logger.list_of_scalars_summary("train","xxx",tensorboard_log, batches_done)

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
        if epoch % opt.checkpoint_interval == 0:
            saveCheckPoint(opt.model_def,{
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
            }, f"checkpoint_%d.pth" % epoch)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary("valid","xxx",evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            # 往tensorboard中记录bn权重分布
            _, bn_weights = gather_bn_weights(model)
            logger.writer.add_histogram('bn_weights/hist', bn_weights.numpy(), epoch, bins='doane')
        stepLR.step()
