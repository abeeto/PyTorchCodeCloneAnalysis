import os
import ssl
import json
import time
import logging
from tqdm import tqdm
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import (ReduceLROnPlateau, CosineAnnealingLR)

import data_loader
from data_loader import transform
from data_loader.dataloader import data_split

from model import classification as model
from torchvision import models
from utils import metrics as metrics
from utils import logger
from utils import custom_loss
from utils import general

import trainer
import test as tester
import argparse

# from torchsampler import ImbalancedDatasetSampler


def main(
        model,
        dataset,
        validation_flag,
        comment="No comment",
        checkpoint=None,
        num_of_class = 2
    ):

    # Checking cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {} ".format(device))

    if checkpoint is not None:
        print("...Load checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("...Checkpoint loaded")
    
    # Convert to suitable device
    model = model.to(device)
    print("Number of parameters: ",sum(p.numel() for p in model.parameters()))
    logging.info("Model created...")

    # using parsed configurations to create a dataset
    data = cfg["data"]["data_csv_name"]
    print("Reading training data from file: ", data)
    training_set = pd.read_csv(data)

    # check if validation flag is on
    if validation_flag == 0:
        # using custom validation set
        print("Creating validation set from file")
        valid = cfg["data"]["validation_csv_name"]
        print("Reading validation data from file: ", valid)
        valid_set = pd.read_csv(valid)
    else:
        # auto divide validation set
        print("Splitting dataset into train and valid....")
        validation_split = float(cfg["data"]["validation_ratio"])
        training_set, valid_set, _, _ = data_split(training_set, validation_split)
        print("Done Splitting !!!")
        
    data_path = cfg["data"]["data_path"]
    batch_size = int(cfg["data"]["batch_size"])
    
    # Create dataset
    training_set = dataset(training_set, data_path, transform.train_transform)
    valid_set = dataset(valid_set, data_path, transform.val_transform)

    # End sampler
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False
    )
    logging.info("Dataset and Dataloaders created")

    # create a metric for evaluating
    # global train_metrics
    # global val_metrics
    train_metrics = metrics.Metrics(cfg["train"]["metrics"])
    val_metrics = metrics.Metrics(cfg["train"]["metrics"])
    print("Metrics implemented successfully")

    # method to optimize the model
    # read settings from json file
    loss_function = cfg["optimizer"]["loss"]
    optimizers = cfg["optimizer"]["name"]
    learning_rate = cfg["optimizer"]["lr"]

    # initlize optimizing methods : lr, scheduler of lr, optimizer
    try:
        # if the loss function comes from nn package
        criterion = getattr(
            nn, loss_function, "The loss {} is not available".format(loss_function)
        )
    except:
        # use custom loss
        criterion = getattr(
            custom_loss,
            loss_function,
            "The loss {} is not available".format(loss_function),
        )
    criterion = criterion()
    optimizer = getattr(
        torch.optim, optimizers, "The optimizer {} is not available".format(optimizers)
    )
    max_lr = 3e-3  # Maximum LR
    min_lr = 1e-5  # Minimum LR
    t_max = 10     # How many epochs to go from max_lr to min_lr
    # optimizer = torch.optim.Adam(
    # params=model.parameters(), lr=max_lr, amsgrad=False)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    save_method = cfg["train"]["lr_scheduler_factor"]
    patiences = cfg["train"]["patience"]
    lr_factor = cfg["train"]["reduce_lr_factor"]
    scheduler = ReduceLROnPlateau(optimizer, mode = save_method, min_lr = min_lr, patience = patiences, factor = lr_factor)
    # scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

    print("\nTraing shape: {} samples".format(len(train_loader.dataset)))
    print("Validation shape: {} samples".format(len(val_loader.dataset)))
    print("Beginning training...")
    
    # export the result to log file
    logging.info("--------------------------------")
    logging.info("session name: {}".format(cfg["session"]["sess_name"]))
    # logging.info(model)
    logging.info("CONFIGS:")
    logging.info(cfg)

    # training models
    num_epoch = int(cfg["train"]["num_epoch"])
    best_val_acc = 0
    t0 = time.time()
    for epoch in range(0, num_epoch):
        t1 = time.time()
        print(('\n' + '%13s' * 3) % ('Epoch', 'gpu_mem', 'mean_loss'))
        train_loss, val_loss, train_result, val_result = trainer.train_one_epoch(
            epoch, num_epoch,
            model, device,
            train_loader, val_loader,
            criterion, optimizer,
            train_metrics, val_metrics, 
        )
        scheduler.step(val_loss)

        # lr scheduling
        logging.info("\n------Epoch %d / %d, Training time: %.4f seconds------" % (epoch + 1, num_epoch, (time.time() - t1)))
        logging.info("Training loss: {} - Other training metrics: {}".format(train_loss, train_result))
        logging.info("Validation loss: {} - Other validation metrics: {}".format(val_loss, val_result))
        
        tb_writer.add_scalar("Training Loss", train_loss, epoch + 1)
        tb_writer.add_scalar("Valid Loss", val_loss, epoch + 1)
        tb_writer.add_scalar("Training Accuracy", train_result["accuracy_score"], epoch + 1)
        tb_writer.add_scalar("Valid Accuracy", val_result["accuracy_score"], epoch + 1)
        # tb_writer.add_scalar("training f1_score", train_result["f1_score"], epoch + 1)
        # tb_writer.add_scalar("valid f1_score", val_result["f1_score"], epoch + 1)
        
        # saving epoch with best validation accuracy
        if best_val_acc < float(val_result["accuracy_score"]):
            logging.info("Validation accuracy= "+ str(val_result["accuracy_score"]))
            logging.info("====> Save best at epoch {}".format(epoch+1))
            best_val_acc = val_result["accuracy_score"]
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss': val_loss,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, log_dir + "/Checkpoint.pt")
        
        
    # testing on test set
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data)

    # prepare the dataset
    testing_set = dataset(test_df, data_path, transform.val_transform)
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=32, shuffle=False)
    print("\nInference on the testing set")

    # load the test model and making inference
    checkpoint = torch.load(log_dir + "/Checkpoint.pt")
    test_model = checkpoint['model']
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)

    # logging report
    report = tester.test_result(test_model, test_loader, device, cfg)
    logging.info("\nClassification Report: \n {}".format(report))
    logging.info('%d epochs completed in %.3f seconds.' % (num_epoch , (time.time() - t0)))

    print("Classification Report: \n{}".format(report))
    print('%d epochs completed in %.3f seconds.' % (num_epoch , (time.time() - t0)))
    print(f'Start Tensorboard with "tensorboard --logdir {log_dir}", view at http://localhost:6006/')
    # # saving torch models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-c', '--configure', default='cfgs/tenes.cfg', help='JSON file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint = args.checkpoint

    # read configure file
    with open(args.configure) as f:
        cfg = json.load(f)
    
    # comment for this experiment: leave here
    comment = cfg["session"]["sess_name"]

    # automate the validation split or not
    if ( float(cfg["data"]["validation_ratio"]) > 0 and cfg["data"]["validation_csv_name"] == ""):
        print("No validation set available, auto split the training into validation")
        validation_flag = cfg["data"]["validation_ratio"]
    else:
        validation_flag = 0

    # choose dataloader type
    module_name = cfg["data"]["data.class"]
    try:
        dataset = getattr(data_loader.dataloader, module_name)
        print("Successfully imported data loader module")

    except:
        print("Cannot import data loader module".format(module_name))

    # # choose model classification
    # module_name = cfg["train"]["model.class"]
    # try:
    #     cls = getattr(model, module_name)
    #     print("Successfully imported model module")
    # except:
    #     print("Cannot import model module".format(module_name))

    cls =  models.resnet50(pretrained=True)
    cls.fc = torch.nn.Linear(cls.fc.in_features,2)
    print("Successfully imported model module")

    # get num of class
    num_of_class = len(cfg["data"]["label_dict"])

    # create dir to save log and checkpoint
    save_path = cfg['train']['save_path']
    time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
    sess_name = cfg["session"]["sess_name"]
    log_dir = general.make_dir_epoch_time(save_path, sess_name, time_str)
    
    # create logger
    log_file = logger.make_file(log_dir, 'result.txt')
    logger.log_initilize(log_file)
    tb_writer = logger.make_writer(log_dir = log_dir)
    logging.info(f'Start Tensorboard with "tensorboard --logdir {log_dir}", view at http://localhost:6006/')
    print("All checkpoint will be saved to {}".format(log_dir))
    print("Done Loading!!!\n")
    
    main(
            model=cls,
            dataset=dataset,
            validation_flag=validation_flag,
            comment=comment,
            checkpoint=checkpoint,
            num_of_class=num_of_class,
        )