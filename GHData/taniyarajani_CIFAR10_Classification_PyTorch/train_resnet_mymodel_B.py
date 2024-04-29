"""
  IDS576: Machine Learning Application with Python
 
  - Filename: train.py
  - Reference: Problem 1 
  - Comments:
      Your job at this code is to understand high-level structure of the assignment, 
      verifying how to load the data, and 
"""

## For pretty print of tensors.
## Must be located at the first line except the comments.
from __future__ import print_function

## Import the basic modules.
import argparse
import numpy as np

## Import the PyTorch modules.
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

## Import the custom module to load the CIFAR-10 dataset.
## (This is not a built-in module, but included in cifar10.py)
from cifar10 import CIFAR10

## You are supposed to implement the following four source codes:
## {softmax.py, twolayernn.py, convnet.py, mymodel.py}.
import models.softmax 
import models.twolayernn
import models.convnet
import models.mymodel

import models.resnet_mymodel_B

# To visualize wrong images
import matplotlib.pyplot as plt

## Initilize a command-line option parser.
parser = argparse.ArgumentParser(description='CIFAR-10 Example')

## Add a list of command-line options that users can specify.
## Shall scripts (.sh) files for specifying the proper options are provided. 
parser.add_argument('--lr', type=float, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', help='number of epochs to train')
parser.add_argument('--model', choices=['softmax', 'convnet', 'twolayernn', 'mymodel', 'resnet_mymodel_B'], help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int, help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int, help='size of convolution kernels/filters')

## Add more command-line options for other configurations.
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='number of batches between logging train status')
parser.add_argument('--cifar10-dir', default='data', help='directory that contains cifar-10-batches-py/ (downloaded automatically if necessary)')

## Parse the command-line option.
args = parser.parse_args()

## CUDA will be supported only when user wants and the machine has GPU devices.
args.cuda = not args.no_cuda and torch.cuda.is_available()

## Change the random seed.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

## Set the device-specific arguments if CUDA is available.
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

## Initialize size information for CIFAR10 dataset
## CIFAR-10 consists of 32x32 color images with 3 channels.
## Each image is one of the ten classes: {airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}.
im_size = (3, 32, 32)   
n_classes = 10

## Normalize each image by subtracdting the mean color and divde by standard deviation.
## For convenience, per channel mean color and standard deviation are provided.
cifar10_mean_color = [0.49131522, 0.48209435, 0.44646862]
cifar10_std_color = [0.01897398, 0.03039277, 0.03872553]

## Define a standardized transform by using PyTorch's transforms module.
## Just one preparation can be used for every split of dataset.
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(cifar10_mean_color, cifar10_std_color),
            ])

## Load training, validation, and test data separately.
## (CIFAR10 class inherits PyTorch Dataset class to load datafile on disk)
## Apply the normalizing transform uniformly across three dataset.
train_dataset = CIFAR10(args.cifar10_dir, split='train', download=True, transform=transform)
val_dataset = CIFAR10(args.cifar10_dir, split='val', download=True, transform=transform)
test_dataset = CIFAR10(args.cifar10_dir, split='test', download=True, transform=transform)

## DataLoaders provide various ways to get batches of examples.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

## Load the proper neural network model.
if args.model == 'softmax':
    # Problem 2 (no hidden layer, input -> output)
    model = models.softmax.Softmax(im_size, n_classes)
elif args.model == 'twolayernn':
    # Problem 3 (one hidden layer, input -> hidden layer -> output)
    model = models.twolayernn.TwoLayerNN(im_size, args.hidden_dim, n_classes)
elif args.model == 'convnet':
    # Problem 4 (multiple hidden layers, input -> hidden layers -> output)
    model = models.convnet.CNN(im_size, args.hidden_dim, args.kernel_size, n_classes)
elif args.model == 'mymodel':
    # Problem 5 (multiple hidden layers, input -> hidden layers -> output)
    model = models.mymodel.MyModel(im_size, args.hidden_dim, args.kernel_size, n_classes)
elif args.model == 'resnet_mymodel_B':
    # Problem 5 (multiple hidden layers, input -> hidden layers -> output)
    model = models.resnet_mymodel_B.Combine(args.hidden_dim, args.kernel_size, n_classes)
else:  
    raise Exception('Unknown model {}'.format(args.model))

## Deinfe the loss function as cross-entropy.
## This is the softmax loss function (i.e., multiclass classification).
criterion = functional.cross_entropy

## Activate CUDA if specified and available.
if args.cuda:
    model.cuda()

#############################################################################
# TODO: Initialize an optimizer from the torch.optim package using the      #
# appropriate hyperparameters found in args. This only requires one line.   #
#############################################################################

optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


## Function: train the model just one iteration.
def train(epoch):
    # Recall model is a class that inherits nn.Module that we learned in the class.
    # This puts the model in train mode as opposed to eval mode, so it knows which one to use.
    model.train()
    
    # For each batch of training images,
    for batch_idx, batch in enumerate(train_loader):
        # Read images and their target labels in the current batch.
        images, targets = Variable(batch[0]), Variable(batch[1])
        
        # Load the current training example in the CUDA core if available.
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.      #
        # This only requires a couple lines of code.                                #
        #############################################################################

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        # Print out the loss and accuracy on the first 4 batches of the validation set.
        # You can adjust the printing frequency by changing --log-interval option in the command-line.
        if batch_idx % args.log_interval == 0:
            # Compute the average validation loss and accuracy.
            val_loss, val_acc = evaluate('val', n_batches=4)
            
            # Compute the training loss.
            train_loss = loss.data.item()
            
            # Compute the number of examples in this batch.
            examples_this_epoch = batch_idx * len(images)

            # Compute the progress rate in terms of the batch.
            epoch_progress = 100. * batch_idx / len(train_loader)

            # Print out the training loss, validation loss, and accuracy with epoch information.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))

## Function: evaluate the learned model on either validation or test data.
def evaluate(split, verbose=False, n_batches=None):
    # Recall model is a class that inherits nn.Module that we learned in the class.
    # This puts the model in eval mode as opposed to train mode, so it knows which one to use.
    model.eval()

    # Initialize cumulative loss and the number of correctly predicted examples.  
    loss = 0
    correct = 0
    n_examples = 0

    # Load the correct dataset between validation and test data based on the split option.
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader

    incorr = []
    wrong_predict = []

    # For each batch in the loaded dataset,
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):        
            data, target = batch
            # Load the current training example in the CUDA core if available.
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                
            # Read images and their target labels in the current batch.
            data, target = Variable(data), Variable(target)
            
            # Measure the output results given the data.
            output = model(data)
    
            # Accumulate the loss by comparing the predicted output and the true target labels.
            loss += criterion(output, target, reduction='sum').data
            
            # Predict the class by finding the argmax of the log-probabilities among all classes.
            pred = output.data.max(1, keepdim=True)[1]
    
            # Add the number of correct classifications in each class.
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            if split == 'test':
                # incorrect identification for part (d) - analyzing incorrect images
                wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
                
                wrong_preds = pred[wrong_idx]
                
                                
                # if incorrect == False:
                #     print("lets see!")
                #     wrong_predict.append(pred.item())
    
                # indx = (incorrect == False).nonzero()
                
            
                list = [element.item() for element in wrong_idx.flatten()]
                list1 = [element.item() for element in wrong_preds.flatten()]
                
                incorr.append(list)
                wrong_predict.append(list1)

            # Keep track of the total number of predictions.
            n_examples += pred.size(0)
    
            # Skip the rest of evaluation if the number of batches exceed the n_batches.
            if n_batches and (batch_i >= n_batches):
                break
    
    # Compute the average loss per example.
    loss /= n_examples

    # Compute the average accuracy in terms of percentile.
    acc = 100. * correct / n_examples

    if split == 'test':
        
        flat_list = [val for sublist in incorr for val in sublist]
        Wpred_list = [val for sublist in wrong_predict for val in sublist]
        

        
        # subsetting incorrect images
        img = test_dataset.test_data[flat_list, :]
        
        label = [test_dataset.test_labels[i] for i in flat_list]
    
        img = img.reshape(len(img), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    
        fig, axes1 = plt.subplots(8,8,figsize=(10,10))
        plt.suptitle('Wrongly classified image')
        for j in range(8):
            for k in range(8):
                i = np.random.choice(range(len(img)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(img[i:i+1][0])
                axes1[j][k].set_title('True: %i and Pred %i' % (label[i], Wpred_list[i]), fontsize = 7.5)
           
        
        fig.savefig('Bad_Images_%s.png'%args.model) 


    # If verbose is True, then print out the average loss and accuracy.
    if verbose:        
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))

    print("The accuracy is %0.2f"%acc)
    return loss, acc

if __name__ == '__main__':
    ## Train the model one epoch at a time.
    for epoch in range(1, args.epochs + 1):
        train(epoch)

    ## Evaluate the model on the test data and print out the average loss and accuracy.
    ## Note that you should use every batch for evaluating on test data rather than just the first four batches.
    evaluate('test', verbose=True)

    ## Save the model (architecture and weights)
    torch.save(model, args.model + '.pt')


    """
    # Later you can call torch.load(file) to re-load the trained model into python
    # See http://pytorch.org/docs/master/notes/serialization.html for more details
    """
