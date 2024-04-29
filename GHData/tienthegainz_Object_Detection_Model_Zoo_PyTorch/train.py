import argparse
import collections
import numpy as np
import os

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from datasets.dataloader import collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from datasets.dataloader import VOCDataset
from torch.utils.data import DataLoader
from eval import evaluate

parser = argparse.ArgumentParser(
    description='Simple training script ')

parser.add_argument('--model', default='retinanet', choices=['retinanet', 'centernet', 'faster-r-cnn'],
                    type=str, help='VOC or COCO')

parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', help='Dataset root directory path')

parser.add_argument(
    '--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
parser.add_argument('--epochs', help='Number of epochs',
                    type=int, default=100)

parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-s', '--isize', default=256, type=int,
                    help='Image size')

parser.add_argument('--save_folder', default='./saved/weights/', type=str,
                    help='Directory for saving checkpoint models')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retina_main():
    # Create the data loaders
    train_dataset = VOCDataset(args.dataset_root,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    valid_dataset = VOCDataset(args.dataset_root,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)

    # Create the model
    if args.depth == 18:
        retinanet = model.retina_bb_resnet18(
            num_classes=train_dataset.num_classes(), pretrained=True)
    elif args.depth == 34:
        retinanet = model.retina_bb_resnet34(
            num_classes=train_dataset.num_classes(), pretrained=True)
    elif args.depth == 50:
        retinanet = model.retina_bb_resnet50(
            num_classes=train_dataset.num_classes(), pretrained=True)
    elif args.depth == 101:
        retinanet = model.retina_bb_resnet101(
            num_classes=train_dataset.num_classes(), pretrained=True)
    elif args.depth == 152:
        retinanet = model.retina_bb_resnet152(
            num_classes=train_dataset.num_classes(), pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        retinanet.load_state_dict(checkpoint['state_dict'])

    retinanet.to(device)
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    init_loss = 1000

    try:
        os.mkdir(args.save_folder)
    except Exception as err:
        print(err)
    try:
        os.mkdir(os.path.join(args.save_folder, args.dataset))
    except Exception as err:
        print(err)
    try:
        os.mkdir(os.path.join(args.save_folder, args.dataset,
                              "retina_bb_resnet{}".format(args.depth)))
    except Exception as err:
        print(err)

    for epoch in range(start_epoch, args.epochs):
        retinanet.train()
        retinanet.freeze_bn()
        classification_loss = None
        regression_loss = None

        retinanet.training = True

        epoch_loss = []

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            image = data['img'].to(device)
            annot = data['annot'].to(device)
            classification_loss, regression_loss = retinanet([image, annot])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                print('Loss reach 0')
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

        print(
            'Epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
        del classification_loss
        del regression_loss
        if epoch % 3 == 0:
            retinanet.training = False
            if args.dataset == 'VOC':
                evaluate(valid_dataset, retinanet)
        scheduler.step(np.mean(epoch_loss))
        state = {
            'epoch': epoch,
            'state_dict': retinanet.state_dict(),
            'num_classes': train_dataset.num_classes(),
        }
        if np.mean(epoch_loss) < init_loss:
            init_loss = np.mean(epoch_loss)
            torch.save(
                state,
                os.path.join(
                    args.save_folder,
                    args.dataset,
                    "retina_bb_resnet{}".format(args.depth),
                    "checkpoint_{}.pth".format(epoch)
                )
            )

    print('Training finished...\n')


if __name__ == "__main__":
    if args.model == 'retinanet':
        retina_main()
    else:
        print('Not supported yet')
