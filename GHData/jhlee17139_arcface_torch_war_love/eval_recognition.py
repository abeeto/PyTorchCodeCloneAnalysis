import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.face_recognition import get_face_recognition_model
from dataset import get_test_single_dataloader
from lr_scheduler import PolyScheduler
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


def main(args):
    # get config
    cfg = get_config(args.config)
    os.makedirs(cfg.output, exist_ok=True)

    test_loader = get_test_single_dataloader(
        cfg.test_rec,
        cfg.test_batch_size,
        cfg.seed,
        cfg.num_workers
    )

    labels_name = cfg.labels

    recognition_model = get_face_recognition_model(cfg)
    recognition_model.load_state_dict(torch.load(cfg.train_weight))
    recognition_model.cuda()
    recognition_model.eval()

    with torch.no_grad():
        total_cnt = 0
        total_accuracy = 0

        labels_length = len(labels_name)
        labels_correct = list(0. for i in range(labels_length))
        labels_total = list(0. for i in range(labels_length))
        per_label_accuracy = list(0. for i in range(labels_length))

        for test_idx, (img, local_labels) in enumerate(test_loader):
            img = img.cuda()
            local_labels = local_labels.cuda()
            outputs = recognition_model(img)
            _, predicted = torch.max(outputs, 1)

            total_cnt += outputs.size(0)
            total_accuracy += (predicted == local_labels).sum().item()

            # per class eval
            label_correct_running = (predicted == local_labels).squeeze()

            label_list = local_labels.cpu().tolist()
            label_correct_list = label_correct_running.cpu().tolist()

            for correct_idx in range(len(label_list)):
                if label_correct_list[correct_idx]:
                    labels_correct[label_list[correct_idx]] += 1

                labels_total[label_list[correct_idx]] += 1

            '''
            label = local_labels[0]
            if label_correct_running.item():
                labels_correct[label] += 1
            labels_total[label] += 1
            '''

        mean_accuracy = total_accuracy / total_cnt
        print('total accuracy : {}'.format(mean_accuracy))

        for i in range(labels_length):
            per_label_accuracy[i] = labels_correct[i] / labels_total[i]
            print('labels {} accuracy : {}'.format(labels_name[i], per_label_accuracy[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Recognition Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
