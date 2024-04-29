'''
Author: Ambareesh Ravi
Description: To visualize conv 2d activations for CNNs in pytorch
Date: Jul 10, 2020
'''
import torch
import torch.nn
import matplotlib.pyplot as plt
from matplotlib import gridspec

# from conv_models import Model_v1

class ClassActivationMaps:
    def __init__(self, model, layer_names, layers_to_hook):
        # get the model for which visualization is to be performed
        self.model = model
        
        # this stores the activations for each run batch_size x map_channels x map_height x map_width
        self.activations = dict()
        
        # give some names to the layer
        # ['conv1', 'bn1', 'relu1', 'conv2', 'bn2', 'relu2', 'conv3', 'bn3', 'relu3', 'conv4', 'bn4', 'relu4', 'conv5', 'bn5', 'relu5']
        self.layer_names = layer_names 
        self.layers = dict([(layer_name, layer) for layer_name, layer in zip(self.layer_names, list(self.model))])
        self.layers_to_hook = layers_to_hook
        
        self.make_hooks()
    
    def get_activation(self, name):
        '''
        Creates and registers forward hooks to get activation for every forward pass
        '''
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def make_hooks(self):
        '''
        Registers a hook for each of the layers in the network
        '''
        for idx, (layer_name, layer) in enumerate(zip(self.layer_names, list(self.model))):
            if layer_name in self.layers_to_hook:
                self.model[idx].register_forward_hook(self.get_activation(layer_name))
    
    def get_activations(self):
        '''
        Returns activations for every layer as a dict
        '''
        return self.activations
    
def activations2grid(original_image, activations, save_path, prefix, resize_to = (128,128), figsize = (15, 15), cmap_type = 'viridis',  pad_inches_border = 0.2):
    '''
    Args:
        original_image - image as <np.array>
        activations - layer names and activations as <dict>
        save_path - path to save the images as a <str>
        prefix - <str> as prefix for saving file
        resize_to - resolution to which images are to be resized to as <tuple>
        figsize - figure size as <tuple>
        cmap_type - type of the color map to which the gray scale has to be converted to
        pad_inches_border - border width around the image
    Returns:
        -
    Exception:
        -
    '''
    for layer_idx, (layer, activation) in enumerate(activations.items()):
        try:
            activation = activation.squeeze().detach().numpy()
            c, w, h = activation.shape
            grid_size = int(np.sqrt(c)), int(np.ceil(np.sqrt(c)))
            if grid_size[0] < 2 or grid_size[1] < 2: grid_size = (2,2)
            fig, ax = plt.subplots(nrows = grid_size[0], ncols = grid_size[1], figsize=figsize, sharex=True, sharey=True)
            gs1 = gridspec.GridSpec(grid_size[0], grid_size[1])
            gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
            idx = 0
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    try:
                        act_map = activation[idx, :, :]
                        act_map = np.uint8(255 * act_map)
                        overlay = cv2.cvtColor(cv2.applyColorMap(act_map, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
                        base = cv2.resize(original_image, resize_to)
                        overlay = cv2.resize(overlay, resize_to)
                        if len(base.shape) < 3: base = np.array([base]*3).transpose((1,2,0))
#                         overlay = cv2.resize(overlay, tuple(list(original_image.shape)[:2]))
                        image_to_display = cv2.addWeighted(base, 0.7, overlay, 0.5, 0.05)
                        ax[row][col].imshow(image_to_display)
                        ax[row][col].axis('off')
                        idx += 1
                    except Exception as e:
                        ax[row][col].axis('off')
                        continue
            
            fig.savefig(os.path.join(save_path, "%d_%s_%s.png"%(layer_idx+1, prefix, layer)), bbox_inches = 'tight', pad_inches = pad_inches_border)        
            fig.clf()
            plt.close()
            
            averaged = np.mean(activation, axis = 0)
            averaged_act_map = np.uint8(255 * averaged)
            overlay = cv2.cvtColor(cv2.applyColorMap(averaged_act_map, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
            base = cv2.resize(original_image, resize_to)
            overlay = cv2.resize(overlay, resize_to)
            if len(base.shape) < 3: base = np.array([base]*3).transpose((1,2,0))
            image_to_display = cv2.addWeighted(base, 0.7, overlay, 0.5, 0.05)
            
            im = Image.fromarray(image_to_display)
            im.save(os.path.join(save_path,"%s_%s_averaged.png"%(prefix, layer)))
            
        except Exception as e:
            continue
            
def visualize_activations(image_samples, save_path):
    '''
    Visualizes the activation for a list of image samples
    
    Args:
        image_samples - list of image input tensors
        save_path - path to save <str>
    Returns:
        -
    Exception:
        -
    '''
    for image_sample in tqdm(image_samples):
        input_tensor = get_image_tensor(PIL_image) # rewrite this function to convert image to preprocessed tensor
        outputs = model(input_tensor.reshape(1,3,128,128)) # declare the model before and change input resolution
        
        activations = cam.activations
        activations2grid(image_sample, activations, save_path, "model_v1")
        cam.activations = dict()
        
        del input_tensor, outputs, encoder_activations, decoder_activations

def visualize_weights(model):
    '''
    visualizes the weights in the network
    
    Args:
        model - loaded pytorch model
    Returns:
        -
    Exception:
        -
    '''
    for layer in list(model.main): # change to model.main to layer name / sequential block name
        try:
            weights = layer.weight
            if len(weights.shape) < 3: continue
            print(layer)
            weights = torch.sum(weights, axis = 1).detach().numpy() # sum for channels
            c, w, h = weights.shape
            grid_size = int(np.sqrt(c)), int(np.ceil(np.sqrt(c)))
            fig, ax = plt.subplots(nrows = grid_size[0], ncols = grid_size[1], figsize=(15,15))
            idx = 0
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    try:
                        ax[row][col].imshow(weights[idx, :, :], cmap = 'jet')
                        ax[row][col].axis('off')
                        idx += 1
                    except Exception as e:
                        ax[row][col].axis('off')
                        continue
            plt.show()
        except Exception as e:
            continue
