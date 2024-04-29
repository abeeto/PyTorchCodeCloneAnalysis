#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: p
# DATE CREATED:                                   
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following 3 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##


# Imports python modules
import argparse


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='flower_data/',
                         help='Images directory')
    parser.add_argument('--arch', type=str, default='resnet',
                         help='Model Architecture')
    parser.add_argument('--process', type=str, default='train',
                         help='Train or test the model')
    parser.add_argument('--learn_rate', type=str, default='0.001',
                         help='Model Learning Rate')
    parser.add_argument('--layers', type=str, default='5',
                         help='number of hidden units')
    parser.add_argument('--epochs', type=str, default='30',
                         help='training epochs')
    parser.add_argument('--gpu', type=str, default='on',
                         help='Use GPU or CPU')

    args = parser.parse_args()
    args.dir = args.dir+'/' if not args.dir.endswith('/') else args.dir
    
    return args