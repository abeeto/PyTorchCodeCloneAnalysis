# Imports here
import time
import torch
from torch import nn, optim
import helper
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from get_input_args import get_input_args

# --------------------------------------------------------
# args input
args = get_input_args()
dir = args.dir
arch = args.arch
process = args.process
learn_rate = float(args.learn_rate)
layers = int(args.layers)
epochs = int(args.epochs)
gpu = args.gpu

# Models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()

# Models dict
models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
# apply model to input
model = models[arch]
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learn_rate)
# --------------------------------------------------------

# Load Data Function
def load_data(dir):
    train_dir = dir + 'train/'
    valid_dir = dir + 'valid/'
    test_dir = dir + 'test/'

    train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    # define the dataloaders
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)

    with open('cat_to_name.json', 'r') as f:
        names = json.load(f)
    
    Labels_df = pd.DataFrame.from_dict({'Id':names.keys(), 'flower_name':names.values()})
    
    return train_loader, validation_loader, test_loader, Labels_df

# Validation Function
def validate(model, validation_data, criterion):    

    # model.eval() is a kind of switch for some specific
    # layers/parts of the model that behave differently 
    # during training and inference (evaluating) time.
    model.eval()
    
    Validation_Loss = 0
    Validation_Corrects = 0

    # Turn off gradients for validation
    with torch.no_grad():
        for image, label in validation_data:
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            image, label = image.to(device), label.to(device) 
            
            # Forward pass
            outputs = model(image)

            # Calculating Loss
            loss = criterion(outputs, label)
            Validation_Loss += loss.item()

            # Calculating Accuracy
            _, preds = torch.max(outputs.data, 1)
            Validation_Corrects += (preds == label).sum().item()

        Validation_Loss = np.round(Validation_Loss/len(validation_data), decimals=6)
        Validation_Accuracy = np.round((100.*Validation_Corrects/len(validation_data)), decimals=2)
        
    return Validation_Loss, Validation_Accuracy, Validation_Corrects

# Training Function
def train(model, train_data, validation_data, epochs, optimizer, criterion):    

    # Epochs or loops number
    Epochs = epochs+1
    Start_time = time.time()
    
    Training_Loss_list = []
    Validation_Loss_list = []
    Validation_Accuracy_list = []
    Validation_Corrects_list = []
    
    print('Training Started...')

    for epoch in range(1,Epochs):
        Training_Loss = 0
        Training_Accuracy = 0
        epoch_start = time.time()

        for image, label in train_data:
            
            # Checking if GPU is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            image, label = image.to(device), label.to(device) # add this line
            
            # model.train() tells your model that you are training the model. 
            # So effectively layers like dropout, batchnorm etc. 
            # which behave different on the train and test procedures
            model.train()
            
            ### Training pass
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(image)

            # Backward pass
            loss = criterion(output, label)
            loss.backward()

            # Update weights
            optimizer.step()
            
            # Calculating Loss
            Training_Loss += loss.item()
            Training_Loss = np.round(Training_Loss/len(train_data), decimals=6)
            epoch_time = np.round((time.time()-epoch_start)/60, decimals=2)
        
        else:
            # Validation Pass
            Validation_Loss, Validation_Accuracy, Validation_Corrects = validate(model, validation_data)
            Validation_Accuracy = np.round((100.*Validation_Corrects/len(validation_data)), decimals=2)
            
            Training_Loss_list.append(Training_Loss)
            Validation_Loss_list.append(Validation_Loss)
            Validation_Accuracy_list.append(Validation_Accuracy)
            Validation_Corrects_list.append(Validation_Corrects)

            print("# Epoch: {},  Took: {}mins".format(epoch,epoch_time))
            print("# Training Loss: {}, # Correctly Classified: {}, # Samples Number: {}".format(Training_Loss,Validation_Corrects, len(validation_data)))
            print("# Validation Loss: {},  # Validation Accuracy: {}% ".format(Validation_Loss,Validation_Accuracy),'\n','-'*15)
            
            
            
    End_time = time.time()
    print("Total Training Time: {}".format(np.round((End_time-Start_time)/60, decimals=2)))
    print('## Training Done ##')

    return Training_Loss_list, Validation_Loss_list, Validation_Accuracy_list, Validation_Corrects_list

# Plot stats
def plot_stats(epochs, Training_Loss, Validation_Loss, Validation_Accuracy):
    eochs = np.arange(1,epochs+1)
    # Validation Accuracy plots
    plt.figure(figsize=(15,5));
    plt.plot(eochs,[value/100 for value in Validation_Accuracy], color='green', label='Validation Accuracy');

    # Validation loss plots
    plt.plot(eochs,Validation_Loss, color='red', label='Validation Loss');

    # Training loss plots
    plt.plot(eochs,Training_Loss, color='orange', label='Training Loss');

    plt.xlabel('Epochs');
    plt.ylabel('Accuracy & Losses');
    plt.legend();
    plt.savefig('training_stats.png')
    plt.show();

# Save model Function
def save_model(model, path, epochs=epochs, arch=arch):
    # saving the model
    torch.save({'Epoch': epochs,
                'Classifier_state': model.state_dict(),
                'Optimizer_state': optimizer.state_dict(),
                'Loss_Function': criterion,
                }, path+'/model_'+arch+'.pth')


def classifier(img_path, model_name):
    # load the image
    img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 


