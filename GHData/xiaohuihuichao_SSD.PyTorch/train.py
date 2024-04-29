import os
import time
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from config import config
from net_model import SSD
from loss import MultiBoxLoss
from data_set import detection_dataset, collate_fn

def save_model(net, path):
    state = net.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state, path)
    
def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt)
    return rt / dist.get_world_size()

def get_parse():
    parser = argparse.ArgumentParser(description="Simple training script for training ssd.")

    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training.")
    
    parser.add_argument("--data_file", default="label.txt")
    parser.add_argument("--classes_file", default="classes.txt")
    
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of batch_size")
    parser.add_argument("--lr", type=float, default=1e-5, help="lr")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")

    parser.add_argument("--loss_lamda", type=float, default=0.5, help="loss_lamda")

    parser.add_argument("--batch_num_log", type=int, default=100, help="batch_num_log")

    parser.add_argument("--batch_num_save", type=int, default=1000, help="batch_num_save")
    parser.add_argument("--save_path", type=str, default="out_model", help="save_path")
    parser.add_argument("--model", type=str, default="model_file/batch.pt", help="model")
    
    parser.add_argument("--log_path", type=str, default="train.log", help="log_path")
    
    parser = parser.parse_args()
    return parser


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py
def main():
    parser = get_parse()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(parser.local_rank)

    # Create the data loaders
    dataset = detection_dataset(parser.data_file, parser.classes_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=parser.batch_size, num_workers=parser.num_workers, collate_fn=collate_fn, pin_memory=True, sampler=sampler)
    print(f"num_clses:{dataset.num_classes()}, num_data: {len(dataset)}")

    # Create the model
    ssd = SSD(dataset.num_classes()+1)
    if parser.model is not None and os.path.isfile(parser.model):
        print("Loading model.")
        # ssd.load_state_dict(torch.load(parser.model))
        d = collections.OrderedDict()
        checkpoint = torch.load(parser.model)
        for key, value in checkpoint.items():
            tmp = key[7:]
            d[tmp] = value
        ssd.load_state_dict(d)
    else:
        print(f"{parser.model} 不存在")

    config["cuda"] = config["cuda"] and torch.cuda.is_available()
    if config["cuda"]:
        ssd = torch.nn.parallel.DistributedDataParallel(ssd.cuda(), device_ids=[parser.local_rank])

    mbox_loss = MultiBoxLoss(config)

    optimizer = optim.Adam(ssd.parameters(), lr=parser.lr, weight_decay=parser.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=0.5, threshold=1e-3)

    ssd.train()
    for epoch_num in range(1, parser.epochs+1):
        epoch_loss = []

        t = time.time()
        for iter_num, data in enumerate(dataloader, 1):
            optimizer.zero_grad()

            img_tensor, boxes_tensor = data["img"], data["boxes"]
            if config["cuda"]:
                img_tensor = img_tensor.cuda(non_blocking=True)
                boxes_tensor = boxes_tensor.cuda(non_blocking=True)

            predictions = ssd(img_tensor)
            loc_loss, conf_loss = mbox_loss(predictions, boxes_tensor)
            loss = loc_loss*parser.loss_lamda + conf_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssd.parameters(), 0.1)
            optimizer.step()

            reduce_conf_loss = reduce_tensor(conf_loss.data)
            reduce_loc_loss = reduce_tensor(loc_loss.data)
            reduce_loss = reduce_conf_loss + reduce_loc_loss

            if parser.local_rank == 0:
                pre_t = t
                t = time.time()
                text = f"[Epoch: {epoch_num}|{parser.epochs}  Iteration: {iter_num}]" \
                    + f"  conf_loss: {reduce_conf_loss.item():1.4f}  loc_loss: {reduce_loc_loss.item():1.4f}"  \
                    + f"  loss: {reduce_loss.item():1.4f}" \
                    + f"  time:{(t-pre_t)*1000:.1f}ms"
                print(text)

                if iter_num % parser.batch_num_log == 0:
                    with open(parser.log_path, "a", encoding="utf-8") as f:
                        f.write(text+"\n")

                if iter_num % parser.batch_num_save == 0:
                    save_model(ssd, "{}/batch.pt".format(parser.save_path))

                epoch_loss.append(float(reduce_loss.data))

            if iter_num % 200 == 0:
                torch.cuda.empty_cache()
        

        if parser.local_rank == 0:
            save_model(ssd, f"{parser.save_path}/epoch.pt")
            scheduler.step(np.mean(epoch_loss))
            print(f"epoch_mean_loss:{np.mean(epoch_loss):.4f}")

    ssd.eval()
    if parser.local_rank == 0:
        save_model(ssd, f"{parser.save_path}/model_final.pt")


if __name__ == "__main__":
    main()
