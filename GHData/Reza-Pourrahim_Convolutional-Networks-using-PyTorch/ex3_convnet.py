import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        




#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512]
num_epochs = 10
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
norm_layer = None #norm_layer = 'BN'
print(hidden_size)

#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Choose the right data augmentation transforms with the right       #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

data_aug_transforms.extend([
                          transforms.ColorJitter(brightness=(0.0,2.0), contrast=(0.0,2.0), saturation=(0.0,2.0), hue=(-0.5,0.5)), #Randomly Change the brightness, contrast, saturation and hue of the image
                          transforms.Grayscale(3), #return image in RGB
                          transforms.RandomVerticalFlip(p=1.0),
                          transforms.RandomRotation(20)

])

print("Data Augmentation Transforms", data_aug_transforms)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=test_transform
                                          )

#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)







def make_layers(layers, hidden_layers,drop_out, batch_norm=False):#drop_out
    in_channels = 3
    for v in hidden_layers:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.MaxPool2d(kernel_size=2, stride=2),nn.Dropout(p=drop_out), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(inplace=True)]
        in_channels = v
    layers += [nn.Linear(512,10)]
    ## Adding Dropout
    #layers +=[nn.Dropout(p=0.5)]
    return nn.Sequential(*layers)

#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes,drop_out,norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        self.layers = make_layers(layers, hidden_layers, drop_out, batch_norm=True)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.layers[:-1](x)
        out = torch.flatten(out, 1)
        out = self.layers[-1](out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out









class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model_sz = sum(p.numel() for p in model.parameters())
    if disp:
        print(f'Number of Parameters: {model_sz}')
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz




#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image of stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    conv1 = model.layers[0].weight.data.cpu().numpy() 
    for i in range(6):
        plt.figure(figsize = (8,0.5))
        for j in range(16):         
            plt.subplot(i+1,16,j+1); plt.imshow(conv1[i*6+j, ...], vmin=0, vmax=255)
            plt.axis('off')
            plt.axis("tight")
        plt.show()  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------

# Build a model for different Dropout parameters 0.1-0.9
Droupout_ValidationResult = []
for p in np.arange(0.1,1,0.1):
  model = ConvNet(input_size, hidden_size, num_classes, p, norm_layer=norm_layer).to(device)
  # Q2.a - Initialize the model with correct batch norm layer

  model.apply(weights_init)
  # Print the model
  print(model)
  # Print model size
  #======================================================================================
  # Q1.b: Implementing the function to count the number of trainable parameters in the model
  #======================================================================================
  PrintModelSize(model)
  #======================================================================================
  # Q1.a: Implementing the function to visualize the filters in the first conv layers.
  # Visualize the filters before training
  #======================================================================================
  VisualizeFilter(model)



  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

  # Train the model
  lr = learning_rate
  total_step = len(train_loader)
  loss_train = []
  loss_val = []
  best_accuracy = None
  accuracy_val = []
  best_model = type(model)(input_size, hidden_size, num_classes, p, norm_layer=norm_layer) # get a new instance
  #best_model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer)
  es = EarlyStopping(patience=5, mode="max")

  for epoch in range(num_epochs):

      model.train()

      loss_iter = 0
      for i, (images, labels) in enumerate(train_loader):
          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          loss_iter += loss.item()
          
          if (i+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
              
      loss_train.append(loss_iter/(len(train_loader)*batch_size))

      
      # Code to update the lr
      lr *= learning_rate_decay
      update_lr(optimizer, lr)
      
          
      model.eval()
      with torch.no_grad():
          correct = 0
          total = 0
          loss_iter = 0
          for images, labels in val_loader:
              images = images.to(device)
              labels = labels.to(device)
              
              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              
              loss = criterion(outputs, labels)
              loss_iter += loss.item()
          
          loss_val.append(loss_iter/(len(val_loader)*batch_size))

          accuracy = 100 * correct / total
          accuracy_val.append(accuracy)
          print('Validation accuracy is: {} %'.format(accuracy))
          #################################################################################
          # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
          # the model with the best validation accuracy so-far (use best_model).          #
          #################################################################################

          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          es(accuracy, model, 'pytorch_model.bin')   
          if es.early_stop:
              print("Early stopping")             
              break

          # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

      

  # Test the model
  # In test phase, we don't need to compute gradients (for memory efficiency)
  model.eval()



  plt.figure(2)
  plt.plot(loss_train, 'r', label='Train loss')
  plt.plot(loss_val, 'g', label='Val loss')
  plt.legend()
  plt.show()

  plt.figure(3)
  plt.plot(accuracy_val, 'r', label='Val accuracy')
  plt.legend()
  plt.show()



  #################################################################################
  # TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
  # best model so far and perform testing with this model.                        #
  #################################################################################
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  model = ConvNet(input_size, hidden_size, num_classes,p,norm_layer=norm_layer).to(device)
  model.load_state_dict(torch.load(f"pytorch_model.bin"))
  model.eval()
  Droupout_ValidationResult.append(es.best_score)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  #Compute accuracy on the test set
  with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          if total == 1000:
              break

      print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))



  # Q1.c: Implementing the function to visualize the filters in the first conv layers.
  # Visualize the filters before training
  VisualizeFilter(model)



  # Save the model checkpoint
  #torch.save(model.state_dict(), 'model.ckpt')

plt.plot([i for i in np.arange(0.1,1, 0.1)], Droupout_ValidationResult)
