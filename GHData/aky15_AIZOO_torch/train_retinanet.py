import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet_utils import model
from retinanet_utils.dataloader import AIZOODataset
from retinanet_utils.dataloader import Resizer,collater, AspectRatioBasedSampler, Augmenter,  Normalizer
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet_utils import coco_eval
from retinanet_utils import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)
    
    train_path = './train'
    val_path = './val'
    BATCH_SIZE = 8
    num_classes = 3

    dataset_train = AIZOODataset(train_path, transforms=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = AIZOODataset(val_path, transforms=transforms.Compose([Normalizer(), Resizer()]))

    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
            collate_fn=collater)

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,
            collate_fn=collater)
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        
        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, 'retinanet_model/retinanet_{}.pt'.format(epoch_num))
        
    retinanet.eval()

    torch.save(retinanet, 'retinanet_model/model_final.pt')


if __name__ == '__main__':
    main()
