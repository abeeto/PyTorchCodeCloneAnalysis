import os
import ssl
import json
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
from torch.optim.lr_scheduler import (ReduceLROnPlateau, CosineAnnealingLR)
from models import classification as model

from utils import logger
from utils import metrics as metrics
from utils import callbacks
from utils.general import (model_loader,  get_optimizer, get_loss_fn, make_dir_epoch_time)
from data_loader.dataloader import data_split, get_data_loader

import trainer
import test as tester
from torchvision import models
import argparse

# from torchsampler import ImbalancedDatasetSampler

def main(
        model,
        config = None,
        comment="No comment",
        checkpoint=None,
    ):            
    if checkpoint is not None:
        print("...Load checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("...Checkpoint loaded")

    # Checking cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {} ".format(device))

    # Convert to suitable device
    model = model.to(device)
    print("Number of parameters: ",sum(p.numel() for p in model.parameters()))
    logging.info("Model created...")

    # using parsed configurations to create a dataset
    num_of_class = len(cfg["data"]["label_dict"])

    # Create dataset
    train_loader, valid_loader, test_loader = get_data_loader(cfg)
    logging.info("Dataset and Dataloaders created")

    # create a metric for evaluating
    train_metrics = metrics.Metrics(cfg["train"]["metrics"])
    val_metrics = metrics.Metrics(cfg["train"]["metrics"])
    print("Metrics implemented successfully")

    # read settings from json file
    # initlize optimizing methods : lr, scheduler of lr, optimizer
    learning_rate = cfg["optimizer"]["lr"]
    optimizer = get_optimizer(cfg)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_fn(cfg)
    criterion = loss_fn()
    
    ## Learning rate decay
    max_lr = 3e-3  # Maximum LR
    min_lr = cfg["optimizer"]["min_lr"]  # Minimum LR
    t_max = 10     # How many epochs to go from max_lr to min_lr
    save_method = cfg["optimizer"]["lr_scheduler_factor"]
    lr_patiences = cfg["optimizer"]["lr_patience"]
    lr_factor = cfg["optimizer"]["reduce_lr_factor"]
    scheduler = ReduceLROnPlateau(optimizer, mode = save_method, min_lr = min_lr, patience = lr_patiences, factor = lr_factor)
    # scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

    print("\nTraing shape: {} samples".format(len(train_loader.dataset)))
    print("Validation shape: {} samples".format(len(valid_loader.dataset)))
    print("Beginning training...")
    
    # export the result to log file
    logging.info("--------------------------------")
    logging.info("session name: {}".format(cfg["session"]["sess_name"]))
    # logging.info(model)
    logging.info("CONFIGS:")
    logging.info(cfg)

    # initialize the early_stopping object
    checkpoint_path = os.path.join(log_dir,"Checkpoint.pt")
    save_mode = cfg["train"]["mode"]
    early_patience = cfg["train"]["early_patience"]
    early_stopping = callbacks.EarlyStopping(patience=early_patience, mode = save_mode, path = checkpoint_path)

    # training models
    num_epoch = int(cfg["train"]["num_epoch"])
    best_val_acc = 0
    t0 = time.time()

    for epoch in range(num_epoch):
        t1 = time.time()
        train_loss, train_acc, val_loss, val_acc, train_result, val_result = trainer.train_one_epoch(
            epoch, num_epoch,
            model, device,
            train_loader, valid_loader,
            criterion, optimizer,
            train_metrics, val_metrics, 
        )

        train_checkpoint = {
            'epoch': epoch + 1,
            'valid_loss': val_loss,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        scheduler.step(val_loss)

        ## lr scheduling
        logging.info("\n------Epoch %d / %d, Training time: %.4f seconds------" % (epoch + 1, num_epoch, (time.time() - t1)))
        logging.info("Training loss: {} - Other training metrics: {}".format(train_loss, train_result))
        logging.info("Validation loss: {} - Other validation metrics: {}".format(val_loss, val_result))
        
        ## tensorboard
        tb_writer.add_scalar("Training Loss", train_loss, epoch + 1)
        tb_writer.add_scalar("Valid Loss", val_loss, epoch + 1)
        tb_writer.add_scalar("Training Accuracy", train_result["accuracy_score"], epoch + 1)
        tb_writer.add_scalar("Valid Accuracy", val_result["accuracy_score"], epoch + 1)
        # tb_writer.add_scalar("training f1_score", train_result["f1_score"], epoch + 1)
        # tb_writer.add_scalar("valid f1_score", val_result["f1_score"], epoch + 1)
        
        # Save model
        if save_mode == "min":
            early_stopping(val_loss, train_checkpoint)
        else:
            early_stopping(val_acc, train_checkpoint)
        if early_stopping.early_stop:
            logging.info("Early Stopping!!!")
            break

    # testing on test set
    # load the test model and making inference
    print("\nInference on the testing set")
    checkpoint = torch.load(checkpoint_path)
    test_model = checkpoint['model']
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)

    # logging report
    report = tester.test_result(test_model, test_loader, device, cfg)
    logging.info("\nClassification Report: \n {}".format(report))
    logging.info('Completed in %.3f seconds.' % (time.time() - t0))

    print("Classification Report: \n{}".format(report))
    print('Completed in %.3f seconds.' % (time.time() - t0))
    print('Start Tensorboard with tensorboard --logdir {}, view at http://localhost:6006/'.format(log_dir))
    # # saving torch models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-c', '--configure', default='cfgs/tenes.json', help='JSON file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint = args.checkpoint

    # read configure file
    with open(args.configure) as f:
        cfg = json.load(f)
    
    # comment for this experiment: leave here
    comment = cfg["session"]["sess_name"]

    # create dir to save log and checkpoint
    save_path = cfg['train']['save_path']
    time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
    sess_name = cfg["session"]["sess_name"]
    log_dir = make_dir_epoch_time(save_path, sess_name, time_str)
    
    # create logger
    log_file = logger.make_file(log_dir, 'result.txt')
    logger.log_initilize(log_file)
    tb_writer = logger.make_writer(log_dir = log_dir)
    logging.info(f"Start Tensorboard with tensorboard --logdir {log_dir}, view at http://localhost:6006/")
    print("--------All checkpoint will be saved to ```{}```--------".format(log_dir))

    cls_model = model_loader(cfg)
    print("Successfully imported model module")
    print("Done Loading!!!\n")
    time.sleep(2.7)
    
    main(
        config = cfg,
        model=cls_model,
        comment=comment,
        checkpoint=checkpoint,
    )