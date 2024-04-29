import os
import sys
import pdb
import glob
import tqdm
import torch
import losses
import argparse
import numpy as np

from utils import utils
from validate import Validation
from models import lstm, transformer, graph
from dataloaders.tfrecord_dataset import TFRecordFrameDataSet
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
# disable tensorflow deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--vocab-size', type=int, default=3862)
parser.add_argument('--train-dir', type=str, required=True, help="directory to save models")
parser.add_argument('--train-data-pattern', type=str, default='/data2/yt8m/v2/frame/train*.tfrecord')
parser.add_argument('--val-data-pattern', type=str, default='/data/yt8m/v3/frame/validate*.tfrecord')
parser.add_argument('--model', type=str, required=True, help="model to use for training")
parser.add_argument('--lr-decay', type=float, default=0.8)
parser.add_argument('--lr-decay-steps', type=int, default=4000000)
parser.add_argument('--base-lr', type=float, default=3e-4)
parser.add_argument('--segment-labels', type=bool, required=True)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--num-features', type=list, default=[1024, 128])
parser.add_argument('--features', type=list, default=['rgb', 'audio'])
parser.add_argument('--num-epochs', type=int, default=5)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--weight-decay', type=float, default=1e-8)
parser.add_argument('--validate-every', type=int, default=-1)
parser.add_argument('--validate-after-epoch', type=bool, default=False)
parser.add_argument('--top-n', type=int, default=10000)
parser.add_argument('--validate-file', type=str, default='/data/yt8m/v3valid.csv')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--save-dir', type=str, default='data/gcn_conv_model_run2')


def get_val_data_loader(args, validate_file=None):
    if validate_file:
        lines = open(validate_file, 'r')
        val_files = [l.strip() for l in lines]
        files = glob.glob(args.val_data_pattern)
        files = [f for f in files if f.split('/')[-1] in val_files]
        print("Len of val files:", len(files))
    else:
        files = glob.glob(args.val_data_pattern)

    record_dataset = TFRecordFrameDataSet(files, segment_labels=args.segment_labels)
    data_loader = torch.utils.data.DataLoader(record_dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=record_dataset.get_collate_fn(),
                                              num_workers=args.num_workers)
    return data_loader


def get_train_data_loader(args, validate_file=None):
    if validate_file:
        lines = open(validate_file, 'r')
        val_files = [l.strip() for l in lines]
        files = glob.glob(args.train_data_pattern)
        print("Before Len:", len(files))
        files = [f for f in files if f.split('/')[-1] not in val_files]
        print("total Len:", len(files))
    else:
        files = glob.glob(args.train_data_pattern)
    record_dataset = TFRecordFrameDataSet(files, segment_labels=args.segment_labels)
    data_loader = torch.utils.data.DataLoader(record_dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=record_dataset.get_collate_fn(),
                                              num_workers=args.num_workers)
    return data_loader


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, amsgrad=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.baser_lr, amsgrad=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def train_loop(model, data_loader, val_obj, args, device):
    optimizer = get_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    step_count = 0
    best_map = 0
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs-1))
        print('-' * 10)

        # setting model to training mode
        model = model.train()
        model = model.double()
        model = model.to(device)

        running_loss = 0.0
        for batch_sample in data_loader:
            batch_video_matrix = batch_sample["video_matrix"].to(device)
            batch_num_frames = batch_sample["video_num_frames"].to(device)
            batch_label_weights = batch_sample["label_weights"].to(device)
            batch_segment_labels = batch_sample["segment_labels"].to(device)
            A_norm = utils.get_feature_similarity_matrix(batch_video_matrix, batch_num_frames, device, normalize=True)

            optimizer.zero_grad()
            output = model(batch_video_matrix, batch_num_frames, A_norm)
            loss = losses.cross_entropy_loss(output, batch_segment_labels, batch_label_weights)
            loss.backward()
            optimizer.step()

            print("Epoch: {}, Step Count: {}, Batch Loss: {}".format(epoch, step_count, loss.data))
            step_count += 1

            if args.validate_every > 0 and (step_count % args.validate_every) == 0:
                print("Running validation")
                with torch.no_grad():
                    mean_ap, _, _ = val_obj.validation_pipeline(top_n=args.top_n)
                    if mean_ap > best_map:
                        best_map = mean_ap
                        print("Saving Model")
                        utils.save_model(model, args.save_dir, save_file="model", step_count=step_count)

            if args.save_every > 0 and (step_count % args.save_every) == 0:
                print("Saving Model")
                utils.save_model(model, args.save_dir, save_file="model", step_count=step_count)

        if args.validate_after_epoch:
            print("Running validation")
            with torch.no_grad():
                val_obj.validation_pipeline(top_n=args.top_n)


def get_feature_similarity(data_loader, device):
    for batch_sample in data_loader:

        batch_vids = batch_sample["video_ids"]
        batch_video_matrix = batch_sample["video_matrix"]
        batch_num_frames = batch_sample["video_num_frames"]

        A_norm = utils.get_feature_similarity_matrix(batch_video_matrix, batch_num_frames, device, normalize=True)
        aggregation = torch.bmm(A_norm, batch_video_matrix)

        aggregation_sim = utils.get_feature_similarity_matrix(aggregation, batch_num_frames, device, normalize=True)

        import matplotlib.pyplot as plt

        for i in range(len(A_norm)):
            print('http://data.yt8m.org/2/j/i/' + batch_vids[i][0][:2] + '/' + batch_vids[i][0] + '.js')
            plt.subplot(2, 1, 1)
            plt.imshow(A_norm[i])
            plt.subplot(2, 1, 2)
            plt.imshow(aggregation_sim[i])
            # pdb.set_trace()
            plt.show()


def get_val_obj(args, model, device):
    val_data_loader = get_val_data_loader(args, validate_file=args.validate_file)
    val_obj = Validation(val_data_loader, model, device, label_cache="/data/yt8m/v3_label_cache")
    return val_obj


def main():
    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)

    # model_class = utils.find_class_by_name(args.model, [graph])

    model = graph.GraphModel(vocab_size=args.vocab_size)
    data_loader = get_train_data_loader(args, args.validate_file)
    val_obj = get_val_obj(args, model, device)
    train_loop(model, data_loader, val_obj, args, device)

    # get_feature_similarity(data_loader, args)


if __name__ == '__main__':
    main()