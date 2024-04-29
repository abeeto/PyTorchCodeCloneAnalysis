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
import torch.optim as optim

from tensorboardX import SummaryWriter
writer = SummaryWriter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--use_gpu", default=True, help="use gpu or not")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    opt = parser.parse_args()
    print(opt)

    print_parser(opt)

    logger = Logger("logs")
    os.makedirs("checkpoints", exist_ok=True)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not opt.use_gpu:
        device = torch.device("cpu")
        print("using CPU!")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("using GPU!")
        else:
            device = torch.device("cpu")
            print("user setting is use GPU, but cuda is not available, using CPU!")

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    ############################################# training one epoch #############################################
    for epoch in range(opt.epochs):
        print("\n\nstart training epoch[{}].\n".format(epoch))

        model.train()

        start_time = time.time()

        #train_epoch_labels = []
        #train_epoch_sample_metrics = []  # List of tuples (TP, confs, pred)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            # # Extract labels
            # targets = targets.cpu()
            # train_epoch_labels += targets[:, 1].tolist()
            # # Rescale target
            # targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            # targets[:, 2:] *= opt.img_size
            # outputs = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
            # train_epoch_sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=opt.iou_thres)

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # # Tensorboard logging, maybe not compatble with all tf version
                # tensorboard_log = []
                # for j, yolo in enumerate(model.yolo_layers):
                #    for name, metric in yolo.metrics.items():
                #        if name != "grid_size":
                #            tensorboard_log += [(f"{name}_{j+1}", metric)]
                # tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            writer.add_scalar('train/mloss-to-batch', loss.item(), batches_done)

            # Determine approximate time left for epoch, ETA estimated time arrival
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            
            print(log_str)

            model.seen += imgs.size(0)

        # # Concatenate sample statistics
        # true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*train_epoch_sample_metrics))]
        # precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, train_epoch_labels)
        # writer.add_scalar('train/precision',precision.mean(), epoch)
        # writer.add_scalar('train/recall',recall.mean(), epoch)
        # writer.add_scalar('train/mAP',AP.mean(), epoch)
        # writer.add_scalar('train/f1',f1.mean(), epoch)
        # for i, c in enumerate(ap_class):
        #     writer.add_scalar('train/'+class_names[c]+"-AP", AP[i], epoch)
        ######################################### END training one epoch #########################################


        ############################################### Evaluation ###############################################
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
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            writer.add_scalar('evaluate/precision',precision.mean(), epoch)
            writer.add_scalar('evaluate/recall',recall.mean(), epoch)
            writer.add_scalar('evaluate/mAP',AP.mean(), epoch)
            writer.add_scalar('evaluate/f1',f1.mean(), epoch)
            
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            print(ap_class)
            # ap_class is a list which AP is calculated for
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                writer.add_scalar('evaluate/'+class_names[c]+"-AP", AP[i], epoch)
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
        ############################################### END Evaluation ###############################################
        
        ############################################### save checkpoint ##############################################
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
        ############################################# END save checkpoint #############################################

writer.close()