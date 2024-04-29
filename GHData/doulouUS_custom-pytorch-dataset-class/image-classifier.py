import ast
import argparse
import logging
import time

import os
import copy

# import numbers

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel

import torch.optim
from torch.optim import lr_scheduler

import torch.utils.data
import torch.utils.data.distributed
# import torchvision
from torchvision import models
from torchvision import transforms
# import torch.nn.functional as F

from custom_transforms import RandomCrop, RandomHorizontalFlip
from custom_datasets import MultiNpzDataset

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ('class1', 'class2')


def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    # 1. Initialize the distributed environment.
    if is_distributed:
    
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    # 2. Loading dataset
    logger.info("Loading Image dataset")
    data_transforms = {
        'train': transforms.Compose([
            RandomCrop(224),  # custom transform that work on numpy arrays!
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        i: MultiNpzDataset(
                data_dir=path,
                transform=data_transforms[i]
            )
        for i, path in [("train", args.train_data), ("val", args.val_data)]
    }

    image_dataloaders = {
        torch.utils.data.DataLoader(
                image_datasets[i],
                batch_size=4, 
                shuffle=True,
                num_workers=4
            )
        for i in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = classes 


    # 3. Loading model
    logger.info("Model loading")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # Configuring the head on top of the pretrained net
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(class_names))

    logger.info("Done.")

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # 4. Setup loss and optimizer
    logger.info("Setting up loss, optimizer and scheduler.")
    criterion = nn.CrossEntropyLoss()  # no .to(device)?
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)   # Decay LR by a factor of 0.1 every 7 epochs
    logger.info("Done.")
    
    # 5. Start training
    logger.info("Training in progress...")
    model = train_model(
        model, 
        criterion, 
        optimizer, 
        exp_lr_scheduler,
        dataloaders=image_dataloaders["train"],
        num_epochs=25,
        device=device,
        dataset_sizes=dataset_sizes
    )


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def train_model(model, criterion, optimizer, scheduler, dataloaders,num_epochs, device, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, class_names, device, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':

    # Container structure
    # /opt/ml
    # |-- input
    # |   |-- config
    # |   |   |-- hyperparameters.json
    # |   |   `-- resourceConfig.json
    # |   `-- data
    # |       `-- <channel_name>
    # |           `-- <input data>
    # |-- model  # write the result of your runs here, can be a full tree of folders
    # |   `-- <model files>
    # `-- output  # write the reason of failure only
    #     `-- failure


    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist-backend', type=str, default='   ', help='distributed backend (default: gloo)')

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default="foo/")  # ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default="foo/")  #os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default="foo/")  #os.environ['SM_MODEL_DIR'])

    # CHANNELS
    parser.add_argument('--train-data', type=str, default="foo/")  #os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--val-data', type=str, default="foo/")  #os.environ['SM_CHANNEL_VALIDATION'])

    parser.add_argument('--num-gpus', type=int, default=1)  #os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())


    # python container/image-classifier/image-classifier.py \
    #     --workers 2 \
    #     --epochs 5 \
    #     --batch-size 32 \
    #     --lr 0.001 \
    #     --momentum 0.9 
