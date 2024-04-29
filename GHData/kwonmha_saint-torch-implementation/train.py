
import argparse
import os.path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data import KT1Data, SaintDataset
from models.noam import NoamOpt, NoamLR
from utils import get_model

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", type=str, default="saint",
                    choices=["saint", "vsaint", "dkt", "vrnn"])
model_args, _ = parser.parse_known_args()
model = get_model(model_args.model)
parser = model.get_parser(parser)
parser.add_argument("--data_path", type=str, default="/ext_hdd/mhkwon/knowledge-tracing/data/Ednet/KT1-train-all.csv")
parser.add_argument("--save_path", type=str, default="/ext2/mhkwon/knowledge-tracing/saved-models/Ednet/")
parser.add_argument("--min_items", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gpu", type=str, default="1")
parser.add_argument("--norm_first", action="store_true", default=False)
parser.add_argument("--noam", action="store_true", default=False)
parser.add_argument("--mynoam", action="store_true", default=False)
parser.add_argument("--warmup_step", type=int, default=4000)
args = parser.parse_args()

data_set_name = os.path.basename(args.data_path)
args.save_path += args.model + "-"
save_path = args.save_path + data_set_name[:-4] + "_d{}_l{}_dr{}_lr{}_b{}_{}.pt".format(
                        args.model_dim, args.n_layers, args.dropout_prob, args.lr, args.batch_size, args.optimizer)
if args.noam:
    save_path = save_path[:-3] + "-wu{}-noam.pt".format(args.warmup_step)
elif args.mynoam:
    save_path = save_path[:-3] + "-wu{}-mynoam.pt".format(args.warmup_step)

# handle duplicate files with same experiment settings
setting_index = 1
while os.path.isfile(save_path):
    save_path = save_path[:-3] + "-{}.pt".format(setting_index)
    setting_index += 1

print("model will be stored on:", save_path)

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

start = time.time()
train_data = KT1Data(args.data_path, True)
if args.data_path.count('train') > 1:
    sys.exit("path has more than one 'train'.")
# change '*train.csv' into '*valid.csv'
valid_path = args.data_path.replace('train', 'valid')
valid_data = KT1Data(valid_path, True)
print("Loaded Data")
print(time.time() - start)

train_dataset = SaintDataset(train_data.data, max_seq=args.seq_len, min_items=args.min_items)
val_dataset = SaintDataset(valid_data.data, max_seq=args.seq_len, min_items=args.min_items)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          # num_workers=8,
                          shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        # num_workers=8,
                        shuffle=False)

args.total_ex = max(train_data.n_q, valid_data.n_q)
args.total_cat = 7 # (1 ~ 7)
args.total_in = 2 # (O, X)

# model = SaintModel(dim_model=args.h_dim, num_en=args.n_layers, num_de=args.n_layers,
#                    heads_en=args.n_heads, heads_de=args.n_heads, dropout_prob=args.dropout_prob, seq_len=args.max_seq,
#                    total_ex=total_ex, total_cat=total_cat, total_in=total_in,
#                    norm_first=args.norm_first, att_dropout=args.att_dropout).to(device)
model = model(args).to(device)

# sigmoid included in BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
criterion.to(device)

if args.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
elif args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.noam:
    optimizer = NoamOpt(args.model_dim, args.warmup_step, optimizer)
elif args.mynoam:
    scheduler = NoamLR(args.model_dim, args.warmup_step, optimizer)

print("start training")
# Training
top_auc = 0.
for epoch in range(args.epochs):
    model.train()
    train_loss = []
    all_labels = []
    all_outs = []

    start_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        exercise = inputs["qids"].to(device)
        part = inputs["part_ids"].to(device)
        response = inputs["correct"].to(device)

        optimizer.zero_grad()
        outputs = model(exercise, part, response)
        mask = (exercise != 0)
        labels = labels.to(device)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        # scheduler.step()
        loss_value = loss.item()
        train_loss.append(loss_value)

        masked_output = torch.masked_select(outputs, mask)
        label_mask = torch.masked_select(labels, mask)

        all_labels.extend(label_mask.view(-1).data.cpu().numpy())
        all_outs.extend(masked_output.view(-1).data.cpu().numpy())

        if i % 500 == 499:
            print("[%d, %5d] loss: %.3f, %.2f" %
                  (epoch + 1, i + 1, loss_value, time.time() - start_time))

    train_auc = roc_auc_score(all_labels, all_outs)
    train_loss = np.mean(train_loss)

    # test, evaluate with metrics
    model.eval()
    val_loss = []
    all_labels = []
    all_outs = []
    # `with torch.no_grad()` is necessary for saving memory.
    # Or it causes gpu memory allocation error.
    # Maybe it was due to memory overspending caused by cloning embedding
    # (from arshadshk's model).
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            exercise = inputs["qids"].to(device)
            part = inputs["part_ids"].to(device)
            response = inputs["correct"].to(device)

            outputs = model(exercise, part, response)
            mask = (inputs["qids"] != 0).to(device)
            labels = labels.to(device)

            loss = criterion(outputs, labels.float())
            val_loss.append(loss.item())

            masked_output = torch.masked_select(outputs, mask)
            labels = torch.masked_select(labels, mask)

            # calc auc, acc
            all_labels.extend(labels.view(-1).data.cpu().numpy())
            all_outs.extend(masked_output.view(-1).data.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_outs)
        val_loss = np.mean(val_loss)

        # save best performing model
        print("val, top auc", val_auc, top_auc)
        if val_auc > top_auc:
            top_auc = val_auc
            print("saved best model")
            torch.save(model.state_dict(), save_path)

        print("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(
            epoch + 1, train_loss, train_auc, val_loss, val_auc, time.time() - start_time))



