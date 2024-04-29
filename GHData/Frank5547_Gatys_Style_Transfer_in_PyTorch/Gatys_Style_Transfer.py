"""
Gatys Style Transfer Replication in PyTorch
Author: Francisco Javier Carrera Arias
Date: 01/28/2019
References:
Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks.
2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
doi:10.1109/cvpr.2016.265
"""
# Import the necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models

# Load and transform images function
def load_and_transform(imagePath, max_size = 400, shape = None):
    image = Image.open(imagePath).convert('RGB')
    
    # large images will be coerced to have a max size of 400 by default.
    # Change if needed
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape   
    # Apply image transformations
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    # Add dimension of the batch size
    image = in_transform(image).unsqueeze(0)
    return image

# Function to undo image normalization
def deNormalize(img):
    # Clone and detach to display image while requires_grad = True
    image = img.to('cpu').clone().detach().numpy()
    image = image.squeeze().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Denormalize the image
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

# Run an image forward through a network and get the feature maps for 
# a given set of layers. Default layers are for VGG-19 matching Gatys et al (2016)
def get_layers(image, model, layers=None):    
    ## Complete mapping layer names of PyTorch's VGG-19 to names from Gatys et al (2016)
    if layers is None:
        layers = {'0': 'conv1_1','5' : 'conv2_1','10' : 'conv3_1',
                  '19': 'conv4_1','21': 'conv4_2','28' : 'conv5_1'}
    
    # Capture the features of the layers of interest
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

# Calculate the Gram Matrix of a given tensor 
def gram_matrix(tensor):
    # Get the depth, height, and width of the Tensor
    ## reshape it to multiply the features for each channel
    _, d, h, w = tensor.size()
    tensor = tensor.view(d,h*w)
    gram = torch.mm(tensor,tensor.t())
    return gram 

# Perform style transfer on a target image by optimizing total loss (content loss*alpha +
# style loss * beta as described by Gatys et al (2016)). This functions has various defaults
# that should work well for any images, but feel free to experiment.
def style_transfer(targetImage,styleGrams,contentFeatures,show = 500,alpha = 1,
                   beta = 1e7,steps = 2000,
                   styleWeights = {'conv1_1': 1.,'conv2_1': 0.8,'conv3_1': 0.5,
                   'conv4_1': 0.3,'conv5_1': 0.1},learning_rate = 0.003):
    
    # Define optimizer
    optimizer = optim.Adam([targetImage], lr = learning_rate)

    for k in range(1, steps+1):
        # Get feature maps from target image, then calculate the content loss
        targetFeatures = get_layers(targetImage,vgg19)
        contentLoss = torch.mean((targetFeatures['conv4_2'] - contentFeatures['conv4_2'])**2)
    
        # Initialize the style loss to 0
        styleLoss = 0
        # Iterate through each style layer and add to the style loss
        for layer in styleWeights:
            # Get the style representation for a layer in the target image
            targetFeature = targetFeatures[layer]
            _, d, h, w = targetFeature.shape
            # Calculate the target's gram matrix of a style layer
            target_gram = gram_matrix(targetFeature)
            # Get the style gram matrix of a given style layer
            styleGram = styleGrams[layer]
            # Calculate the style loss for a layer multiplied by its weight
            layerStyleLoss = torch.mean((target_gram - styleGram)**2) * styleWeights[layer]
            # Add to the style loss
            styleLoss += layerStyleLoss / (d * h * w)
        
        # Calculate the total loss and multiply by alpha and beta parameters
        total_loss = (contentLoss*alpha) + (styleLoss*beta)
    
        # Update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Display intermediate images and print the loss
        if  k % show == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(deNormalize(target))
            plt.axis('off')
            plt.show()
    
# Load the convolutional and pooling layers of the VGG-19 network
# (The fully connected layers of the classifier are not needed)
vgg19 = models.vgg19(pretrained=True).features

# freeze all VGG parameters
for param in vgg19.parameters():
    param.requires_grad_(False)

# Move to CUDA GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19.to(device)

# Declare the images you want to work with. You can give the name of the path to
# the images or just the image file names if they are on your working directory.
contentInput = input("What will be the content image? ")
styleInput = input("What will be the style image? ")

# Load in content and style images
contentImage = load_and_transform(contentInput).to(device)
# Resize style image to match the size of the content image
styleImage = load_and_transform(styleInput, shape=contentImage.shape[-2:]).to(device)

# Get content and style image's features
contentFeatures = get_layers(contentImage, vgg19)
styleFeatures = get_layers(styleImage, vgg19)

# Calculate the gram matrices for each selected layer of our style image
styleGrams = {layer: gram_matrix(styleFeatures[layer]) for layer in styleFeatures}

# Create a target image and allow change Starting with a target image as a copy
# of the content image is a good practice since you can just focus on
# changing its style
target = contentImage.clone().requires_grad_(True).to(device)

# Perform the style transfer procedure
style_transfer(target,styleGrams,contentFeatures,steps = 2500)

# display the difference in the target image with respect to the content image
# after style transfer
fig, axes = plt.subplots(1, 2, figsize=(15, 15))
axes[0].imshow(deNormalize(contentImage))
axes[1].imshow(deNormalize(target))
plt.axis('off')
plt.show()

# Ouput the final target image as a .png Enjoy!
fig = plt.subplots(1,1,figsize = (20,20))
plt.imshow(deNormalize(target))
plt.axis('off')
plt.savefig("Style_transferred.png")

print("Style transfer complete. New image saved as Style_transferred.png" )
