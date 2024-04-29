import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.face_recognition import get_face_recognition_model
from dataset import get_single_dataloader
from lr_scheduler import PolyScheduler
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


def main(args):
    # get config
    cfg = get_config(args.config)
    os.makedirs(cfg.output, exist_ok=True)

    train_loader = get_single_dataloader(
        cfg.classification_rec,
        cfg.batch_size,
        cfg.seed,
        cfg.num_workers
    )
    test_loader = get_single_dataloader(
        cfg.test_rec,
        cfg.test_batch_size,
        cfg.seed,
        cfg.num_workers
    )

    recognition_model = get_face_recognition_model(cfg)
    recognition_model.load_backbone_weight()
    recognition_model.cuda()
    recognition_model.train()

    # class_dist = cfg.labels_dist
    # class_dist = torch.sqrt(1 / torch.tensor(class_dist, dtype=torch.float32))
    # class_dist = class_dist.cuda()
    # class_dist = 1 / torch.tensor(class_dist, dtype=torch.float32)
    # class_dist = nn.functional.softmax(class_dist)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=class_dist, reduction='mean')
    '''
    opt = torch.optim.AdamW(
        params=[{"params": recognition_model.parameters()}, ],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    '''
    opt = torch.optim.SGD(
        params=[{"params": recognition_model.parameters()}, ],
        lr=cfg.lr)

    cfg.total_batch_size = cfg.batch_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    start_epoch = 0
    global_step = 0

    # summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))

    for epoch in range(start_epoch, cfg.num_epoch):
        total_train_loss = 0.0
        total_train_batch = 0

        for train_idx, (img, local_labels) in enumerate(train_loader):
            img = img.cuda()
            local_labels = local_labels.cuda()

            global_step += 1
            output = recognition_model(img)
            loss = criterion(output, local_labels)

            loss.backward()
            opt.step()

            total_train_loss += loss.item()
            total_train_batch = train_idx

            opt.zero_grad()
            # lr_scheduler.step()

        with torch.no_grad():
            total_test_loss = 0.0
            total_test_batch = 0

            for test_idx, (img, local_labels) in enumerate(test_loader):
                img = img.cuda()
                local_labels = local_labels.cuda()
                output = recognition_model(img)
                loss = criterion(output, local_labels)
                total_test_loss += loss.item()
                total_test_batch = test_idx

            mean_train_loss = total_train_loss / total_train_batch
            mean_test_loss = total_test_loss / total_test_batch

            print('{}.epoch : {} train loss, {} test loss'.format(epoch, mean_train_loss, mean_test_loss))
            path_module = os.path.join(cfg.output, "{}_epoch_model.pt".format(epoch))
            torch.save(recognition_model.state_dict(), path_module)

    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(recognition_model.state_dict(), path_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Recognition Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
