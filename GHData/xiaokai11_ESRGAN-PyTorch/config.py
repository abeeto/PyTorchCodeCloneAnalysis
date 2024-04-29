# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn


# testetst
# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
# ----------------------------------------------
# mode = test
# mode = train_esrgan
mode = "train_rrdbnet"
# Experiment name, easy to save weights and log files
# ------------------------------------------------
# exp_name = ESRGAN_baseline
exp_name = "RRDBNet_baseline"

if mode == "train_rrdbnet":
    # Dataset address
    train_image_dir = "./data/DIV2K/ESRGAN/train"
    valid_image_dir = "./data/DIV2K/ESRGAN/valid"
    test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = "./data/Set5/GTmod12"

    image_size = 128
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
# train-rrdbnet---------------
    # pretrained_model_path = "./results/pretrained_models/RRDBNet_x4-DFO2K-2e2a91f4.pth.tar"
    pretrained_model_path =  ""
    # Incremental training and migration training
# resume-rrdbnet ----------------------------------------------------------------------
    # resume = f"./samples/RRDBNet_baseline/g_epoch_xxx.pth.tar"
    resume = f""

    # Total num epochs
    epochs = 108

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    print_frequency = 200

if mode == "train_esrgan":
    # Dataset address
    train_image_dir = "./data/DIV2K/ESRGAN/train"
    valid_image_dir = "./data/DIV2K/ESRGAN/valid"
    test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = "./data/Set5/GTmod12"

    image_size = 128
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model

    pretrained_d_model_path = ""
    # pretrained_g_model_path = "./results/RRDBNet_baseline/g_last.pth.tar"
    pretrained_g_model_path = "./results/RRDBNet_baseline/g_best.pth.tar"

    # Incremental training and migration training

# resume-esrgan--------------------------------------------------------------------
    # resume_d = "samples/ESRGAN_baseline/g_epoch_xxx.pth.tar"
    # resume_g = "samples/ESRGAN_baseline/g_epoch_xxx.pth.tar"
    resume_d = ""
    resume_g = ""

    # Total num epochs
    epochs = 44

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.34"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    adversarial_weight = 0.005

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    print_frequency = 200

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"
    hr_dir = "./data/Set5/GTmod12"
# test---------------------
    model_path = "./results/pretrained_models/RRDBNet_x4-DFO2K-2e2a91f4.pth.tar"
