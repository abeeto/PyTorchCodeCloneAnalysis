'''
Helper functions to be used across the program. This file does not import any other files as modules!
'''
import os
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

############################################################### GENERAL METHODS ###############################################################
def get_device():
    '''
    Checks if GPU is available to be used. If not, CPU is used.
    '''
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_file_exist(file_path, file_name):
    '''
    Checks if a file exists at the given path.
    '''
    if os.path.isfile(file_path + file_name):
        return True

    return False

def create_dir(dir_path):
    '''
    Creates a directory at the given path if the directory does not exist.
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_image(image_path, resized_image_size):
    '''
    Reads and resize  a single image from the given path and returns the image in NumPy array.
    '''
    im_ = cv2.imread(image_path) #read image from path
    im_ = cv2.resize(im_, (resized_image_size, resized_image_size)) #resize image
    im_ = im_/255

    img = np.asarray(im_, dtype=np.float32)
    return img


############################################################### DATA PRE-PROCESSING METHODS ###############################################################
def get_classes(dataset_folder_path):
    '''
    Returns a list of class names (lowercased) from the dataset folder and the total number of classes.
    '''
    # class_names = []

    # for folder in glob.glob(dataset_folder_path + '**'):

    #     folder_name = folder.split('/')[-1].lower()
    #     class_names.append(folder_name)

    #This one-liner is equivalent to above commented lines of code.
    class_names = [folder.split('/')[-1].lower() for folder in glob.glob(dataset_folder_path + '**')]
    class_names.sort()

    return class_names, len(class_names)

def generate_img_label(dataset_path, classes, img_exts):
    '''
    Returns a list of tuples containing the absolute image path and the index of the class.
    '''

    # img_label = []

    # for path in glob.glob(dataset_path + '**', recursive=True):

    #     if path[-3:] in img_exts or path[-4:] in img_exts:

    #         class_name = path.split('/')[-2].lower()
    #         class_index = classes.index(class_name)

    #         img_label.append((path, class_index))

    #This one-liner is equivalent to above commented lines of code.
    img_label = [(path, classes.index(path.split('/')[-2].lower())) for path in glob.glob(dataset_path + '**', recursive=True)
                 if path[-3:] in img_exts or path[-4:] in img_exts]

    return img_label


def generate_training_data(data, resized_image_size):
    '''
    Reads the image path and label from the given data and returns them in numpy array.
    '''

    image_path, label = data

    image_array = read_image(image_path=image_path, resized_image_size=resized_image_size)

    return image_array, label


############################################################### DATA POST-PROCESSING METHODS ###############################################################

def calculate_accuracy(network_output, target):
    '''
    Calculates the overall accuracy of the network.
    '''
    num_data = target.size()[0] #num of data
    network_output = torch.argmax(network_output, dim=1)
    correct_pred = torch.sum(network_output == target)

    accuracy = (correct_pred*100/num_data)

    return accuracy

def plot_graph(epochs, x_label, y_label, title, save_path, *args):
    '''
    Plots a graph using matplotlib using the given parameters and saves the graph at the given path.
    *args has to be a list containing N number of value list, N number of colour and N number of label name.
    '''
    plt.clf() #cleares any existing graph
    x_axis = [i for i in range(epochs)]

    for arg in args:
        plt.plot(x_axis, arg[0], arg[1], label=arg[2])

    plt.title(str(title))
    plt.ylabel(str(y_label))
    plt.xlabel(str(x_label))
    plt.legend()
    plt.savefig(save_path)

def evaluate_class(net_output, classes_list):
    '''
    Given the prediction tensor and the list of classes, returns the predicted class.
    '''
    predicted_index = torch.argmax(net_output, dim=1)
    predicted_class = classes_list[predicted_index[0]]

    return predicted_class
