
# test if it is uploaded to my github
# change 2
"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_
In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__
Quoting these notes,
    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.
These two major transfer learning scenarios look as follows:
-  **Finetuning the convnet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

if __name__ == '__main__':
    cudnn.benchmark = True
    plt.ion()   # interactive mode

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    # The problem we're going to solve today is to train a model to classify
    # **ants** and **bees**. We have about 120 training images each for ants and bees.
    # There are 75 validation images for each class. Usually, this is a very
    # small dataset to generalize upon, if trained from scratch. Since we
    # are using transfer learning, we should be able to generalize reasonably
    # well.
    #
    # This dataset is a very small subset of imagenet.
    #
    # .. Note ::
    #    Download the data from
    #    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
    #    and extract it to the current directory.

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'C:/Users/chend/OneDrive/Documents/Pytorch_study_codes/pets'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    print("image datasets: ",image_datasets)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}

    print("Data loaders:",dataloaders)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('dataset sizes:',dataset_sizes)
    class_names = image_datasets['train'].classes
    print("class names:",class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################################################
    # Visualize a few images
    # ^^^^^^^^^^^^^^^^^^^^^^
    # Let's visualize a few training images so as to understand the data
    # augmentations.

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    #####batch_size = 4
    # Get a batch of training data
    #inputs, classes=dataloaders['train'][:10]
    inputs, classes = next(iter(dataloaders['train']))
    #print("the 1st Inputs:")
    #print(inputs[0])
    print("the 1st Labels:")
    print(classes[0])
    print("all Labels(batch_sieze=4:")
    print(classes)
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.


    def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
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
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    ######################################################################
    # Visualizing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # Generic function to display predictions for a few images
    #

    def visualize_model_predictions(model, num_images=6):
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
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    ####################  Method 1   #################################
    # Finetuning the convnet
    # ----------------------
    #Building Transfer learing model
    # Load a pretrained model and reset final fully connected layer.


    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # print("Finetuning  stage: model_ft.fc.out_features#")
    # print(model_ft.fc.out_features)
    # print("Finetuning stage: model_ft.fc.in_features#")
    # print(num_ftrs)
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, lens(class_name)).
    # model_ft.fc = nn.Linear(num_ftrs, 2) #add an fully connected layer
    #
    # model_ft = model_ft.to(device) # If GPU is available, then the model goes to GPU for run
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #
    # ######################################################################
    # # Train and evaluate
    # # ^^^^^^^^^^^^^^^^^^
    # #
    # # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # # minute.
    # #
    # print()
    # print("Model_Finetuning training ...............")
    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=25)
    #
    # ######################################################################
    # # predictions
    # print("Model_Finetuning training is over ...............")
    # print()
    # print()
    # visualize_model_predictions(model_ft)


    ########################   Method 2   ##############################
    # ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad = False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #
    ####model_ft = models.resnet18(pretrained=True)
    ####num_ftrs = model_ft.fc.in_features
    ####model_ft.fc = nn.Linear(num_ftrs, 2)  # add an fully connected layer
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        #param.requires_grad = True
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    print("ConvNet stage: model_conv.fc.out_features#")
    print(model_conv.fc.out_features)
    print("ConvNet stage: model_conv.fc.in_features#")
    print(num_ftrs)
    model_conv.fc= nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    #optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=4, gamma=0.1)


    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # On CPU this will take about half the time compared to previous scenario.
    # This is expected ass  gradientdon't need to be computed for most of the
    # network. However, forward does need to be computed.
    #
    print("ConvNet as fixed feature extractor")
    print("Model_Conv training ...............")
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=8)

    # ######################################################################

    print("Model_Conv training is over ...............")
    visualize_model_predictions(model_conv)

    plt.ioff()
    plt.show()

    #print(model_conv.state_dict())
    #
    # ######################################################################
    # # Further Learning
    # # -----------------
    # #
    # # If you would like to learn more about the applications of transfer learning,
    # # checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
    # #
    # #
