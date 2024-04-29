from cpc_models.MobileNetV2_Encoder import MobileNetV2_Encoder
from cpc_models.ResNetV2_Encoder import PreActResNetN_Encoder
from cpc_models.WideResNet_Encoder import Wide_ResNet_Encoder

from baseline_models.MobileNetV2 import MobileNetV2
from baseline_models.ResNetV2 import PreActResNetN
from baseline_models.WideResNet import Wide_ResNet

from data.data_handlers import *
from argparser.train_classifier_argparser import argparser

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from tqdm import tqdm


def fwd_pass_extract(x, y):

    # Run the network
    net.eval()
    with torch.no_grad():
        outputs = net(x)

    return outputs


def extract_vectors():
    vector_list, label_list = [], []

    for batch_img, batch_lbl in tqdm(train_loader, dynamic_ncols=True):
    # for batch_img, batch_lbl in tqdm(test_loader, dynamic_ncols=True):
        batch_vectors = fwd_pass_extract(batch_img.to(args.device), batch_lbl.to(args.device))
        batch_vectors = torch.mean(batch_vectors, dim=(1, 2))
        vector_list.extend(batch_vectors.cpu().numpy())

        batch_lbl = batch_lbl.cpu().numpy()
        label_list.extend(batch_lbl.reshape(batch_lbl.size))

    df = pd.DataFrame(vector_list, columns=range(vector_list[0].size))
    df['label'] = label_list

    return df


if __name__ == "__main__":
    args = argparser()
    print(f"Running on {args.device}")

    # Get selected dataset
    if args.dataset == "stl10":
        _, train_loader, test_loader = get_stl10_dataloader(args, labeled=True)
    elif args.dataset == "cifar10":
        _, train_loader, test_loader = get_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        _, train_loader, test_loader = get_cifar100_dataloader(args)
    elif "mnist" in args.dataset:
        _, train_loader, test_loader = get_medmnist_dataloader(args)
    elif args.dataset == "kdr":
        _, train_loader, test_loader = get_kdr_dataloader(args)

    # Define network and optimizer for given train_selection
    if not args.fully_supervised:
        print("Extracting CPC Representations")

        # Load the CPC trained encoder (with classifier layer activated)
        if args.encoder[:6] == "resnet":
            net = PreActResNetN_Encoder(args, use_classifier=False)
        elif args.encoder[:10] == "wideresnet":
            parameters = args.encoder.split("-")
            depth = int(parameters[1])
            widen_factor = int(parameters[2])
            net = Wide_ResNet_Encoder(args, depth, widen_factor, use_classifier=False)
        else: # args.encoder == "mobilenetV2"
            net = MobileNetV2_Encoder(args, use_classifier=False)

        colour = "_colour" if (not args.gray) else ""
        encoder_path = os.path.join("TrainedModels", args.dataset, "trained_encoder")
        encoder_path = f"{encoder_path}_{args.encoder}_crop{args.crop}{colour}_grid{args.grid_size}_{args.norm}Norm_{args.pred_directions}dir_aug{args.cpc_patch_aug}_{args.model_num}{args.model_name_ext}.pt"
        
        net.load_state_dict(torch.load(encoder_path))
        net.to(args.device)
        print(f"Loaded Model:\n{encoder_path}")

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    else:
        print("Training Fully Supervised")

        # Load the network
        if args.encoder[:6] == "resnet":
            net = PreActResNetN(args).to(args.device)
        elif args.encoder[:10] == "wideresnet":
            parameters = args.encoder.split("-")
            depth = int(parameters[1])
            widen_factor = int(parameters[2])
            net = Wide_ResNet(args, depth, widen_factor).to(args.device)
        elif args.encoder == "mobilenetV2":
            net = MobileNetV2(num_classes=args.num_classes).to(args.device)
        else:
            raise Exception("Invalid choice of encoder")
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    try:
        df_path = os.path.splitext(encoder_path)[0] + '_vectors_unshuffled.csv'
        df = extract_vectors()
        df.to_csv(df_path, index=False)
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt")
