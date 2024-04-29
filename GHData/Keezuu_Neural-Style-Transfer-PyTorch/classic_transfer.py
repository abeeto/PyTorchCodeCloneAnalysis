import time


import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import torch
import pprint
import copy

from Layers.NormalizeLayer import NormalizeLayer
from Layers.ContentLayer import ContentLayer
from Layers.StyleLayer import StyleLayer

from resources.constants import *
from resources.utilities import *
# -- CONSTANTS --

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Now to properly implement the style transfer we need to:
# 1. Define the custom content layer that will allow us to compute content loss (ContentLayer.py)

# 2. Define the style content layer that will allow us to compute content loss (StyleLayer.py)
# 2.1 Define the computation of Gram matrix (StyleLayer.py)

# 3. Create normalization layer to ensure that images that are passed
# to the network have the same mean and standard deviation as the ones
# VGG was trained on (NormalizeLayer.py)

# 4. Rebuild the VGG19 by inserting the custom content and style layers
# after chosen layers in original VGG19
# this way we can access the activations values of layers in VGG19
# and compute the style and content losses inside content/style layers

# 5. Define the LBFGS optimizer and pass the input image with gradients
# enabled to it

# 6. Write training function


# 4. Rebuild the network
def rebuild_model(nn_model, content_image, style_image,
                  normalize_mean, normalize_std, content_layers_req, style_layers_req):
    model_copy = copy.deepcopy(nn_model)
    # Define the layer names from which we want to pick activations

    # Create a new model that will be modified version of given model
    # starts with normalization layer to ensure all images that are
    # inserted are normalized like the ones original model was trained on
    norm_layer = NormalizeLayer(normalize_mean, normalize_std).to(device)
    model = nn.Sequential(norm_layer)
    model = model.to(device)
    model.eval()
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    # iterator
    i = 0
    # We need to keep track of the losses in content/style layers
    # to compute the total loss therefore we keep those in a list and return it
    # at the end of the function
    style_layers = []
    content_layers = []
    # Loop over the layers
    for layer in model_copy.children():
        # The layers in vgg are not numerated so we have to add numeration
        # to copied layers so we can append our content and style layers to it
        name = ""
        # Check which instance this layer is to name it appropiately
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "Conv2d_{}".format(i)
        if isinstance(layer, nn.ReLU):
            name = "ReLu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        if isinstance(layer, nn.MaxPool2d):
            name = "MaxPool2d_{}".format(i)
        # Layer has now numerated name so we can find it easily
        # Add it to our model
        model.add_module(name, layer)

        # After adding check if it is a layer after which we should add our content
        # or style layer
        # Check for content layers
        if name in content_layers_req:
            # Get the activations in this layer
            content_activations = model(content_image).detach()
            # Create the content layer
            content_layer = ContentLayer(content_activations)
            # Append it to the module
            model.add_module("ContentLayer_{}".format(i), content_layer)
            content_layers.append(content_layer)
        # Check for style layers
        if name in style_layers_req:
            # Get the style activations in this layer
            style_activations = model(style_image).detach()
            # Create the style layer
            style_layer = StyleLayer(style_activations)
            # Append it to the module
            model.add_module("StyleLayer_{}".format(i), style_layer)
            style_layers.append(style_layer)

    # We don't need any layers after the last style or content layer
    # so we need to delete them from the model
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLayer) or isinstance(model[i], StyleLayer):
            break
    model = model[:(i + 1)]

    pprint.pprint(model)

    return model, content_layers, style_layers


# 5. Define the optimizer
def get_optimizer(input_image):
    """Uses LBFGS as proposed by Gatys himself because it gives best results"""
    return optim.LBFGS([input_image.requires_grad_()])


# 6. Write training function
def style_transfer(nn_model, content_image, style_image, input_image, normalize_mean, normalize_std,
                   content_layers_req, style_layers_req, num_steps=500, style_weight=400000, content_weight=1):
    """Runs the style transfer on input image"""
    # Get the rebuilded model and style and content layers
    model, content_layers, style_layers = rebuild_model(nn_model, content_image, style_image, normalize_mean,
                                                        normalize_std, content_layers_req, style_layers_req)
    # Get the LBFGS optimizer
    model.eval()
    input_image = input_image.to(device)
    lbfgs = get_optimizer(input_image)
    style_layers_factor = 1 / len(style_layers_req)
    # Run the optimizer for num_steps
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            input_image.data.clamp_(0, 1)
            # Zero the gradients from last iteration and
            # forward the image through network
            lbfgs.zero_grad()
            model(input_image)
            style_score = 0
            content_score = 0

            # Compute the style and content stores
            # based on values computed in style/content layers during forward propagation
            for sl in style_layers:
                style_score += sl.loss #* style_layers_factor
            for cl in content_layers:
                content_score += cl.loss

            # As described in the paper, formula nr. 7
            style_score *= style_weight
            content_score *= content_weight

            # Compute loss and propagate it backwards
            loss = style_score + content_score
            loss.backward()

            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
               # show_tensor(input_image, "run {}:".format(run))
            run[0] += 1

            return style_score + content_score

        lbfgs.step(closure)

    input_image.data.clamp_(0, 1)

    return input_image


if __name__ == '__main__':
    # We want to use GPU for this computationally expensive task
    assert torch.cuda.is_available()

    pp = pprint.PrettyPrinter(indent=4)

    # we dont need the last fully connected layers and adaptive avgpool so we copy only CNN part of VGG19
    # We send it to GPU and set it to run in eval() mode as in Style Transfer we won't need
    # to train the network
    model = models.vgg19(pretrained=True).features.to(device).eval()
    pprint.pprint(model)

    # Define after which layers we want to input our content/style layers
    # they will enable us to compute the style and content losses during forward propagation
    content_layers_req = ["Conv2d_5"]  # pick layer near the middle
    style_layers_req = ["Conv2d_1", "Conv2d_2", "Conv2d_3", "Conv2d_4", "Conv2d_5", "Conv2d_6", "Conv2d_7", "Conv2d_8"]

    # VGG19 specific mean and std used to normalize images during it's training
    # We will normalize our images using those same values to ensure best results
    # Change this if other model is loaded
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Load the images as preprocessed tensors
    content_name = ["ogonek.jpg", "ogonek1.png", "ogonek2.jpg", "ogonek3.jpg", "ogonek4.jpg", "ogonek5.png",
                    "jeżol.jpg", "jeżol1.jpg", "jeżol2.jpg", "jeżol3.jpg", "jeżol4.jpg",
                    "jeżol5.jpg", "jeżol6.jpg"]
    style_name = ["starrynight", "fajne", "forest_style", "vcm",  "art3"]
    for cn in content_name:
        for sn in style_name:
            content_tensor = image_loader(IMAGES_PATH+cn)
            style_tensor = image_loader(IMAGES_PATH+"{}.jpg".format(sn))

            # Assert that they're same size
            assert content_tensor.size() == style_tensor.size()

            show_tensor(content_tensor, "Content")
           # save_tensor(content_tensor, content_name)

            show_tensor(style_tensor, "Style")
           # save_tensor(style_tensor, style_name)

            input_tensor = content_tensor.clone()

            start_time = time.time()

            output = style_transfer(model, content_tensor, style_tensor, input_tensor,
                                    mean, std, content_layers_req, style_layers_req, num_steps=300)

            elapsed_time = time.time() - start_time
            show_tensor(output, title="output")
            save_tensor(output, "C-"+cn+"S-"+sn)

            print(time.strftime("Elapsed time %Mm:%Ss", time.gmtime(elapsed_time)))
