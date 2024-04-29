import matplotlib.pyplot as plt
import itertools
import numpy as np
import yaml
import os

def save_result(source, target, epoch, num_save, save_dir, loss_z=None):
    
    target = target.cpu().data.numpy()
    source = source.cpu().data.numpy()

    if loss_z is not None:
        loss_z = loss_z.cpu().data.numpy()
        loss_ind = loss_z.argsort()
        
        target = target[loss_ind]
        source = source[loss_ind]

    num_rows = 2

    fig, ax = plt.subplots(num_rows, num_save, figsize=(2*num_save, 2*2))
    for i, j in itertools.product(range(num_rows), range(num_save)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    
    for i in range(num_rows): 
        if i == 0:
            test_images = target
        elif i == 1:
            test_images = source

        for j in range(num_save):
            ax[i, j].cla()
            if test_images.shape[1] == 1:
                ax[i, j].imshow(test_images[j, 0], cmap='gray')
            else:
                ax[i, j].imshow((test_images[j].transpose(1, 2, 0) + 1) / 2)

    plt.savefig(os.path.join(save_dir, 'epoch_{}.png'.format(epoch)))
    plt.close()

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    evaluation_directory = os.path.join(output_directory, 'evaluation')
    if not os.path.exists(evaluation_directory):
        print("Creating directory: {}".format(evaluation_directory))
        os.makedirs(evaluation_directory)
    return checkpoint_directory, image_directory, evaluation_directory

def get_model_list(dirname):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f))]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name
