 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler 
from torch.utils.data import Subset, DataLoader		# create dataset, dataloader
import torchvision
from torchvision import datasets, models, transforms  # image data, transform to torch tensor format 
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from timeit import default_timer as timer
from datetime import timedelta
import argparse
from platform import python_version
import os
from collections import defaultdict

def show_versions():
  print("Versions...")
  print("python", python_version())
  print("torch", torch.__version__)
  print("torchvision", torchvision.__version__)
  print("matplotlib", matplotlib.__version__)
  print("seaborn", sns.__version__)
  print("sklearn", sklearn.__version__)
  print()

# Function to set device
def set_device(req_device):
  print('Device details...')
  if torch.cuda.is_available() == True:
    count = torch.cuda.device_count()
    print("GPU number of devices:", count)
    print(*["GPU device["+str(x)+"]="+torch.cuda.get_device_name(x) for x in range(count)], sep="\n")
    print("GPU current device:", torch.cuda.current_device())

  # Set device to CPU or GPU if available
  device = torch.device('cuda:0' if torch.cuda.is_available() and not (req_device == 'cpu') else 'cpu')
  print('Requested device:', req_device, ', using device:', device, '\n')
  return device


# Function to combine datasets into dict; split training dataset into train and validation
def train_val_split_test(train, val_size):
  train_idx, val_idx = train_test_split(list(range(len(train))), test_size=val_size)
  datasets = {}
  datasets['train'] = Subset(train, train_idx)
  datasets['val'] = Subset(train, val_idx)
  return datasets

# Define function to train the model
def train_model(model, criterion, optimizer, scheduler, epochs=25):
  start = timer()
  dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val']}  

  # init model state (save) and accuracy, in the end we return the best 
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  print('Training model...')
  print(*['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'], sep='\t')
  print('-'*64)
  loss_acc_dict = defaultdict(list)
  
  for epoch in range(epochs):
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode
    
      # init batch loss
      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        # move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # track history in train, not val
        torch.set_grad_enabled(phase == 'train')
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backpropagate and optimize in train, not val
        if phase == 'train':
          loss.backward()
          optimizer.step()
        
        # update batch statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      
      # step learning rate
      if phase == 'train':
        scheduler.step()

      # update epoch statistics
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      if phase=='train':
        print('{}/{}'.format(epoch, epochs - 1), end='\t')
        loss_acc_dict['train_loss'].append(epoch_loss)
        loss_acc_dict['train_acc'].append(epoch_acc.item())
      else:
        loss_acc_dict['val_loss'].append(epoch_loss)
        loss_acc_dict['val_acc'].append(epoch_acc.item())
      end = '\t\t' if phase=='train' else '\n'
      print('{:.4f}\t\t{:.4f}'.format(epoch_loss, epoch_acc), end=end)
 
      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
  
  # best statistics
  time_elapsed = timer() - start
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}\n'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)

  return model, loss_acc_dict

# Preview dataset by batch
def show_images(dataloader, classes, dataset_name):
  images, labels = next(iter(dataloader))
  plt.figure(figsize=(10,6))
  batch_size = images.shape[0]
  for i in range(batch_size):
    plt.subplot(int(batch_size/10),10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img = images[i].permute(1,2,0) # reshape tensor(3,L,W) to matplotlib shape(L,W,3)
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    img = std * img + mean
    img = torch.clamp(img, min=0, max=1)
    plt.imshow(img, cmap=plt.cm.binary)  
    plt.xlabel(classes[labels[i].item()]) # labels are tensors, get value with item()
    plt.suptitle('Batch of ' + str(batch_size) + ' images from ' + dataset_name, fontsize=16, y=.9)
  plt.show()

# Function to test model
def test_model(model, dataloader, device):
  was_training = model.training
  model.eval()
  correct, total = 0, 0
  
  with torch.no_grad():      # do not allocate memory for gradient calculations on model 
    for inputs, labels in dataloader:
      # move data to device
      inputs = inputs.to(device)
      labels = labels.to(device)
      output = model(inputs)
      
      for idx, i in enumerate(output):
        #print(idx, i, torch.argmax(i), y[idx])
        if torch.argmax(i) == labels[idx]:
          correct += 1
        total += 1
  
  model.train(mode=was_training)
  return (correct, total)

# Visualize test dataset on trained model
def visualize_model(model, dataloader, class_names, device, num_images=6):
  was_training = model.training
  model.eval()
  images_so_far = 0
  fig = plt.figure(figsize=(10,10))

  with torch.no_grad():
    for inputs, labels in dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

      for j in range(inputs.size()[0]):
        images_so_far += 1

        # refactor image to matplotlib
        img = inputs[j].cpu()
        img = img.permute(1,2,0) # reshape tensor(3,L,W) to matplotlib shape(L,W,3)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        img = std * img + mean
        img = torch.clamp(img, min=0, max=1)

        # calculate image result
        expected = class_names[labels[j]]
        inferred = class_names[preds[j]]
        color = 'black' if expected==inferred else 'red'
        probs = torch.nn.functional.softmax(outputs[j], -1)
        prob = str(round(torch.max(probs).item()*100, 1))

        # construct matplotlib figure
        cols = min(num_images, 5)
        ax = plt.subplot(num_images//cols, cols, images_so_far)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(expected)
        plt.xlabel(inferred + " (" + prob + "%)", fontdict={'color': color})
        ax.xaxis.label.set_color(color)
        plt.suptitle('Visualization of trained model on test dataset', fontsize=12)
        plt.imshow(img)

        if images_so_far == num_images:
          model.train(mode=was_training)
          plt.show()
          return
      
    model.train(mode=was_training)
    plt.show()

# Function to show Confusion Matrix
def confusion_matrix(model, dataloader, class_names, device):
  was_training = model.training
  model.eval()

  # Initialize confusion matrix
  cf_matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.uint8)

  # Count expected and inferred results
  for images, labels in dataloader:
    inputs = images.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
      
    for i, img in enumerate(outputs):
      expected = labels[i]
      inferred = torch.argmax(img)
      cf_matrix[expected][inferred] += 1
 
  # Plot
  plt.figure(figsize=(10,6))
  ax = sns.heatmap(cf_matrix, annot=True, 
                 yticklabels=class_names, xticklabels=class_names, fmt='', 
                 linewidths=1, linecolor='k', cmap='Blues')
  ax.set(title="Confusion Matrix Heatmap", xlabel="Predicted", ylabel="Actual",)
  plt.show()
  print('Total test digits:', cf_matrix.sum().item())
  print('Predicted distribution:', cf_matrix.sum(0))
  print('Actual distribution:', cf_matrix.sum(1))
  model.train(mode=was_training)

def execute(model, dataset, device, batch_size):
  model.eval()
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  count, fails = 0, 0
  time_to_dev, time_to_cpu, time_dev, total_time = 0,0,0,0

  with torch.no_grad():
    for images, labels in dataloader:
      # measure times
      start = timer() 
      images = images.to(device)
      labels = labels.to(device)
      time_to_dev += timer() - start

      start = timer()
      output = model(images)
      time_dev += timer() - start

      start = timer()
      output.cpu()
      time_to_cpu += timer() - start

      count += len(output)
      for i, img in enumerate(output):
        expected = labels[i]
        inferred = torch.argmax(img)
        if expected != inferred:
          confidence = round(torch.max(torch.exp(img)).item()*100, 1)
          #print("Inference error: " + str(expected.item()) + "," + str(inferred.item()) + "," + str(confidence))
          fails += 1
    print("Results: " + str(fails) + " errors in " + str(count) + " (" + str(round(fails/count*100,2)) + "%)")
   
    total_time = time_to_dev + time_dev + time_to_cpu
    print('Device:', device, ', total images:', str(count), ', batch_size:', str(batch_size))
    print('To device time  : ' + str(time_to_dev) + ' us, ' + str(round(time_to_dev/total_time*100.0,2))+'%')
    print('Processing time : ' + str(time_dev) + ' us, ' + str(round(time_dev/total_time*100.0,2))+'%')
    print('From device time: ' + str(time_to_cpu) + ' us, ' + str(round(time_to_cpu/total_time*100.0,2))+'%')
    print('Total time      : ' + str(total_time) + ' us, ' + str(round(total_time/total_time*100.0,2))+'%')


def print_model(model, input_size):
  from torchsummary import summary
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model_d = model.to(device)
  summary(model_d, input_size)

if __name__ == '__main__':
  show_versions()

  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-d","--device", help="device type(ex. cpu, cuda, cuda:0)")
  parser.add_argument("-t","--train", help="training mode", action="store_true")
  parser.add_argument("-p","--prod", help="production mode", action="store_true")
  args = parser.parse_args()
  
  # Set device to CPU or GPU
  device = set_device(args.device)

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

  data_dir = 'data/hymenoptera_data'
  image_datasets = {}
  image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
  image_val_test = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

  # Split train -> train+val datasets, put both in dictionary
  datasets = train_val_split_test(image_val_test, 0.30)
  image_datasets['val'] = datasets['val']
  test = datasets['train']
  print("Datasets for classification...")
  print('Dataset sizes(training):', {x: len(image_datasets[x]) for x in ['train', 'val']})
  print('Dataset sizes(test):', len(test))
  class_names = image_datasets['train'].classes
  print('Classes:', class_names)

  if args.train:
    # Create training dataloaders (iterable object over a dataset)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}

    # Show some images to verify dataset and dataloaders are functional
    #show_images(dataloaders['train'], class_names, 'train')
    
    # Show model
    print_model(models.resnet18(pretrained=True), (3,500,250)) # ResNet18 output layer Linear-68 = 1000 nodes
    print()

    # Train model -> fine-tuning transfer learning
    model_ft = models.resnet18(pretrained=True)         # start with pretrained model
    num_ftrs = model_ft.fc.in_features                  # 
    model_ft.fc = nn.Linear(num_ftrs, len(class_names)) # add new output layer 
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_ft, _ = train_model(model_ft, criterion, optimizer, scheduler, epochs=25)

    # Test the fine-tuned transfer-learning model
    print('Testing fine-tuned model...')
    dataloader = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False, num_workers=4)
    correct, total = test_model(model_ft, dataloader, device)
    print("Test accuracy: {:.4f}%".format(100*round(correct/total, 3)))
    visualize_model(model_ft, dataloader, class_names, device, num_images=20)
    confusion_matrix(model_ft, dataloader, class_names, device)
    print()

    # Save model as PyTorch
    checkpoint = {'model_state_dict': model_ft.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint_ft.pth')

    # Train model -> fixed-feature extractor transfer learning
    model_ffe = models.resnet18(pretrained=True)         # start with pretrained model 
    for param in model_ffe.parameters():                 # freeze parameters prevents backprop changing weights
      param.requires_grad = False                
    num_ftrs = model_ffe.fc.in_features                  # get number of Linear layer input features
    model_ffe.fc = nn.Linear(num_ftrs, len(class_names)) # change Linear-68 layer output to 2 nodes (new layers  
                                                         # have random weights & requires_grad=True by default)
    model_ffe = model_ffe.to(device)
    criterion = nn.CrossEntropyLoss()                    # Performs LogSoftMax & NLLLoss on logits for each class
    optimizer_conv = optim.SGD(model_ffe.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model_ffe, _ = train_model(model_ffe, criterion, optimizer_conv, exp_lr_scheduler, epochs=25)

    # Test the fixed-feature extractor transfer-learning model
    print('Testing fixed-feature extractor model...')
    dataloader = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False, num_workers=4)
    correct, total = test_model(model_ffe, dataloader, device)
    print("Test accuracy: {:.4f}%".format(100*round(correct/total, 3)))
    visualize_model(model_ffe, dataloader, class_names, device, num_images=20)
    confusion_matrix(model_ffe, dataloader, class_names, device)  
    print()

    # Save model as PyTorch
    checkpoint = {'model_state_dict': model_ffe.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint_ffe.pth')
  
  if args.prod:
    print("\nExecuting production model...")
    model_exe = models.resnet18(pretrained=True)         # start with pretrained model
    num_ftrs = model_exe.fc.in_features                  # 
    model_exe.fc = nn.Linear(num_ftrs, len(class_names)) # add new output layer 
    model_exe = model_exe.to(device)
    
    for model_file in ['checkpoint_ft.pth','checkpoint_ffe.pth']:
      # Load model as PyTorch
      print("\n" + model_file + " model performance...") 
      checkpoint = torch.load(model_file)
      model_exe.load_state_dict(checkpoint['model_state_dict'])
    
      # Test model
      dataloader = DataLoader(test, batch_size=107, shuffle=False)
      correct, total = test_model(model_exe, dataloader, device)
      print("Accuracy: ", round(correct/total, 3))
       
      # Measure time performance
      execute(model_exe, test, device, 107)

