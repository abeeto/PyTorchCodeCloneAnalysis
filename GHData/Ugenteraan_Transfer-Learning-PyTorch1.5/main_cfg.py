'''
Argument PARSERs.
'''
import argparse

DEFAULTS = {
    'dataset_path': '/dataset/',
    'image_exts' : ['jpg', 'png', 'jpeg', 'bmp'],
    'model_path' : '/torch_model/',
    'model_name' : 'vgg16-torch.pth',
    'save_model_name' : 'vgg16-trained.pth',
    'image_size' : 224,
    'rotation_prob' : 0.5,
    'shear_prob' : 0.5,
    'hflip_prob' : 0.5,
    'vflip_prob' : 0,
    'noise_prob' : 0.5,
    'rotation_range': 60,
    'shear_range' : 0.4,
    'noise_mode' : ['gaussian', 'salt', 'pepper', 's&p', 'speckle'],
    'num_workers' : 4,
    'data_shuffle' : True,
    'batch_size' : 5,
    'epoch' : 100,
    'learning_rate' : 1e-4,
    'learning_rate_decay' : 0.99,
    'plot_graph' : True,
    'class_file' : './names.txt'
}

PARSER = argparse.ArgumentParser()


PARSER.add_argument('--mode', type=str, help='Specify the mode of usage. Either train or eval', required=True)
PARSER.add_argument('--dataset_path', type=str, default=DEFAULTS['dataset_path'], help='Specify the relative path of the training data folder.')
PARSER.add_argument('--image_exts', type=list, default=DEFAULTS['image_exts'], help='A list of accepted image extensions.')
PARSER.add_argument('--model_path', type=str, default=DEFAULTS['model_path'], help='Specify the relative path of the model storing folder.')
PARSER.add_argument('--model_name', type=str, default=DEFAULTS['model_name'], help='Specify the name of the model file to be loaded.')
PARSER.add_argument('--save_model_name', type=str, default=DEFAULTS['save_model_name'], help='Specify the name of the model file to be saved.')
PARSER.add_argument('--image_size', type=int, default=DEFAULTS['image_size'], help='Specify the size of the training images to be resized.')
PARSER.add_argument('--rotation_prob', type=int, default=DEFAULTS['rotation_prob'], help='Specify the probability to decide how likely a training image should be rotated. Set to 0 to disable.')
PARSER.add_argument('--shear_prob', type=float, default=DEFAULTS['shear_prob'], help='Specify the probability to decide how likely a training image should be sheared. Set to 0 to disable.')
PARSER.add_argument('--hflip_prob', type=float, default=DEFAULTS['hflip_prob'], help='Specify the probability of a training image to be horizontally flipped.')
PARSER.add_argument('--vflip_prob', type=float, default=DEFAULTS['vflip_prob'], help='Specify the probability of a training image to be vertically flipped.')
PARSER.add_argument('--noise_prob', type=float, default=DEFAULTS['noise_prob'], help='Specify the probability of a training image to be added with noise.')
PARSER.add_argument('--rotation_range', default=DEFAULTS['rotation_range'], help='Specify the range for an image to be rotated.')
PARSER.add_argument('--shear_range', default=DEFAULTS['shear_range'], help='Specify the range for an image to be sheared.')
PARSER.add_argument('--noise_mode', type=list, default=DEFAULTS['noise_mode'], help='Specify the modes of noise to be added.')
PARSER.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'], help='Specify the number of workers to load/process the dataset.')
PARSER.add_argument('--data_shuffle', type=bool, default=DEFAULTS['data_shuffle'], help='Specify whether if the dataset should be shuffled or not.')
PARSER.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'], help='Specify the batch size.')
PARSER.add_argument('--epoch', type=int, default=DEFAULTS['epoch'], help='Specify the training epoch.')
PARSER.add_argument('--learning_rate', type=float, default=DEFAULTS['learning_rate'], help='Specify the learning rate for the training.')
PARSER.add_argument('--learning_rate_decay', type=float, default=DEFAULTS['learning_rate_decay'],help='Specify the decay rate for the learning rate.')
PARSER.add_argument('--plot_graph', type=bool, default=DEFAULTS['plot_graph'], help='Specify whether if graphs should be generated after the training.')
PARSER.add_argument('--class_file', type=str, default=DEFAULTS['class_file'], help='Specify the path to the .txt file where the name of all the classes are stored line by line.')

ARGS = PARSER.parse_args()

assert ARGS.mode == 'train' or ARGS.mode == 'eval', "Error! Mode must be either 'train' or 'eval'."
