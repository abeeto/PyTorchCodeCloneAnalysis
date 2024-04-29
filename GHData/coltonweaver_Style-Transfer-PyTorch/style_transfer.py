import torch
import torch.optim as optim
from torchvision import transforms, models

import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Using the VGG19 model pretrained on ImageNet data, need only the layers
vgg = models.vgg19(pretrained=True).features

# Freeze the parameters because no need to update model
for param in vgg.parameters():
    param.requires_grad_(False)

# Move VGG model to GPU if available for faster processing.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

def load_transform_image(path, max_size=400, shape=None):
    '''
        Load in an image and transform it to put it through algorithm.
    '''
    image = Image.open(path).convert('RGB')

    # Resize the image to necessary size if it is too big
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape
    
    # Now process the input image
    image_trans = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # Remove hidden alpha channel, keeping just RGB and unsqueeze to flatten it.
    image = image_trans(image)[:3,:,:].unsqueeze(0)

    return image

def im_convert(tensor):
    '''
        Takes in a tensor representing an image, and converts it to a proper Image
    '''

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)

    return Image.fromarray(np.uint8(image * 255))

def get_features(image, model, layers=None):
    '''
        Run the given image through the model and collect
        features at different layers
    '''

    # By default use the layers from 
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

    # Now run the image through the model, and store the necessary layers in features dict to return
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    
    return features

def gram_matrix(tensor):
    '''
        Compute the gram matrix of a given tensor.
    '''
    _, d, h, w = tensor.size()
    # Reshape the 3D Matrix to a 2D matrix of d x (h * w)
    tensor = tensor.view(d, h * w)
    # Multiply the matrix by its transpose to get Gram Matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

def transfer_style(content, style):
    '''
        Takes a given content image and style image to analyze and transfer
        the style of the style image to the content image
    '''
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Calculate the Gram Matrix for each set of style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Create a copy of the content to transfer style to, allow Gradients to be calculated
    target = content.clone().requires_grad_(True).to(device)

    # Now make a list of the style layers to weight, used for style loss.
    # Earlier layers are weighted higher to result in greater style artifacts
    style_weights = {
        'conv1_1': 1.,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    }

    # Define alpha (content loss weight) and beta (style loss weight)
    alpha = 1
    beta = 1e6

    # Create an optimizer with the target image as the focus
    optimizer = optim.Adam([target], lr=0.003)
    update = 400
    epochs = 4000 # Can vary, have had good results with 4000

    for ii in range(1, epochs + 1):

        # Grab the features from the current iteration of the target image
        target_features = get_features(target, vgg)

        # Determine the loss wrt the content image
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0

        for layer in style_weights:
            
            # Use the processed features for this layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape

            # Compute the target gram matrix and grab the style gram matrix
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]

            # Calculate the loss wrt the layer target gram and style gram, multiplied by the layer's weight
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

            # Add the layer's loss to the style_loss and normalize by the dimensions of the layer
            style_loss += layer_style_loss / (d * h * w)
        
        # Calculate total loss according to formula
        total_loss = alpha * content_loss + beta * style_loss

        # Take a step with the optimizer, to change the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % update == 0:
            print("Total loss: {}".format(total_loss))
            print("Epoch {}/{}".format(ii, epochs))
    
    return im_convert(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Takes a given content image and style image and transfers the style of the style image to the content.")
    parser.add_argument('-c', dest='c_path', action='store', required=True, default=None, help='Path to content image.')
    parser.add_argument('-s', dest='s_path', action='store', required=True, default=None, help='Path to style image.')
    args = parser.parse_args()

    content = load_transform_image(args.c_path).to(device)
    style = load_transform_image(args.s_path).to(device)

    processed_image = transfer_style(content, style)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

    ax1.imshow(im_convert(content))
    ax1.set_title('Target Image')
    ax2.imshow(im_convert(style))
    ax2.set_title('Style Image')
    ax3.imshow(processed_image)
    ax3.set_title('Processed Image')

    plt.show()

    processed_image.save('style_out.png')