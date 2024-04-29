# -*- coding: utf-8 -*-
"""
Created on August 2022

Auxiliary functions.

@authors: Nuria Gómez-Vargas
"""

import torch
from collections import Counter
import itertools
import numpy as np
import os


##########################################################################################

# SEED

seed = 16
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################################################################################


def l1(x,y, device):
    """
    It computes the elementwise l1-distance between x and y.

    Parameters
    ----------
    x: torch.Tensor
        vector of features.
    y: torch.Tensor
        vector of features.

    Returns
    -------
    torch.Tensor: elementwise l1-distance(x,y).
    """
    return torch.as_tensor( torch.abs(torch.sub(x,y)), device = device)


def evaluating(set_name, dictionary, train_dict, list_rays, device, net, model_name = None):
    """
    Routine to evaluate a (validation or test) dataset against the support (train set).
    
    Parameters
    ----------
    set_name: str
        Validation or test.
    dictionary: dict
        Dataset.
    train_dict: dict
        Support set.
    list_rays: list
        List with the ID of the individuals.
    device: str
        CPU or GPU device.
    net: torch.NN
        Neural network.
    model_name: str
        Identification for the model according to the hyperparameters used.

    Returns
    -------
    mean_accuracy: float
        Mean accuracy across all skates' evaluation.
    dict_preds: dict
        Dictionary that saves the predictions.
    """

    if set_name == 'test':

        net.to(device)
        bestmodel = torch.load(os.getcwd() + '/' + model_name + '.pt', map_location=torch.device(device))
        net.load_state_dict(bestmodel)

    net.eval()
    with torch.no_grad():
    
        accuracies = []
        dict_preds = {}
        for ray1 in list_rays:
            
            prediction_for_each_image_ofray1 = []
            puntuaciones = []
            for feature1 in dictionary[ray1]:
                #mean prob agains support set
                mean_probs = [torch.Tensor([ net( l1(feature1,feature2, device).to(device) ) for feature2 in train_dict[ray2] ]).mean() for ray2 in list_rays]  #TENGO QUE LAS RAYAS DE MI CONJUNTO VALID CONTRA EL CONJUNTO SOPORTE TRAIN
                puntuaciones.append(mean_probs)
                max_prob = max(mean_probs)
                list_pred = [list_rays[ind] for (ind, mean_prob) in enumerate(mean_probs) if mean_prob == max_prob]
                prediction_for_each_image_ofray1.append(list_pred)
                        
            #majority voting
            counting_preds = Counter( list( itertools.chain.from_iterable(prediction_for_each_image_ofray1) ) )
            final_pred = [k for (k,v) in counting_preds.items() if v == max(counting_preds.values()) ]

            puntuaciones = torch.Tensor(puntuaciones).mean(axis = 0)
                        
            ray_accuracy = int(ray1 in final_pred)/len(final_pred)
            if set_name == 'test':
                print('PREDICTION FOR ', ray1, ': ', final_pred, ". Accuracy: ", ray_accuracy)
            dict_preds[ray1] = (final_pred, puntuaciones)
            accuracies.append(ray_accuracy)

        mean_accuracy = np.array(accuracies).mean()
        if set_name == 'test':
            print('MEAN ', set_name,' ACCURACY: ', mean_accuracy)

    return mean_accuracy, dict_preds
