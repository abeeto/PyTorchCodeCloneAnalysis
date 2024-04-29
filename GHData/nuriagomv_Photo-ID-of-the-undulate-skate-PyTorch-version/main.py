# -*- coding: utf-8 -*-
"""
Created on August 2022

MAIN CODE FOR FISH-ID WITH SIAMESE NEURAL NETWORKS

@authors: Nuria Gómez-Vargas
"""

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.models import resnet50, ResNet50_Weights # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
from architecture import NN
from dataTorch import myDataSet
from images_reader import load_dataset
from aux_funcs import evaluating

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def execution_code(batch_size, lr, momentum):
    """
    All the steps for the execution of the code according to the hyperparameters.

    Parameters
    ----------
    batch_size: int
        Size of the training batch.
    lr: float
        Learning rate for the Stochastic Gradient Descent algorithm.
    momemtum: float
        Momentum for the Stochastic Gradient Descent algorithm.

    Returns
    -------
    test_acc: float
        Mean test accuracy across all individuals.
    """

    total_trainiters = 1000000
    validate_each = 100

    ##########################################################################################

    # SEED

    seed = 16
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = str(batch_size)+'-'+str(lr)+'-'+str(momentum)
    
    device = 'cuda' if bool(torch.cuda.device_count()) else 'cpu'
    print("\nDEVICE: ", device)

    ##########################################################################################

    # LOADING DATASET

    #os.chdir(r"C:\Users\nuria\OneDrive - UNIVERSIDAD DE SEVILLA\Académico\beca IIM-CSIC\AI iGENTAC - Nuria\algoritmo_pytorch")
    os.chdir(r"/mnt/beegfs/ngvargas/algoritmo_pytorch/")
    #os.chdir(r"C:\Users\CdeC\Desktop\AI iGENTAC - Nuria\algoritmo_pytorch")

    load_again = False
    list_rays = ['skate_DESTAC-RUN-20-09', 'skate_DESTAC-RUN-20-10', 'skate_DESTAC-RUN-20-12', 'skate_DESTAC-RUN-20-14', 'skate_DESTAC-RUN-20-17', 'skate_DESTAC-RUN-20-19', 'skate_DESTAC-RUN-20-20', 'skate_DESTAC-RUN-20-25', 'skate_DESTAC-RUN-20-27', 'skate_DESTAC-RUN-20-29', 'skate_DESTAC-RUN-20-30', 'skate_DESTAC-RUN-20-34', 'skate_DESTAC-RUN-20-36', 'skate_DESTAC-RUN-20-37', 'skate_DESTAC-RUN-20-38', 'skate_DESTAC-RUN-20-40', 'skate_IGENTAC-RUN-21-01', 'skate_IGENTAC-RUN-21-02', 'skate_IGENTAC-RUN-21-04', 'skate_IGENTAC-RUN-21-05', 'skate_IGENTAC-RUN-21-06', 'skate_IGENTAC-RUN-21-07', 'skate_IGENTAC-RUN-21-10', 'skate_IGENTAC-RUN-21-12', 'skate_IGENTAC-RUN-21-13', 'skate_IGENTAC-RUN-21-14', 'skate_IGENTAC-RUN-21-15', 'skate_IGENTAC-RUN-21-16', 'skate_IGENTAC-RUN-21-17', 'skate_IGENTAC-RUN-21-18', 'skate_IGENTAC-RUN-21-19', 'skate_IGENTAC-RUN-21-20', 'skate_IGENTAC-RUN-21-21', 'skate_IGENTAC-RUN-21-22', 'skate_IGENTAC-RUN-21-23', 'skate_IGENTAC-RUN-21-24', 'skate_IGENTAC-RUN-21-25', 'skate_IGENTAC-RUN-21-27']

    if load_again:
        #dataset_path = r"C:\Users\nuria\Desktop\datasetsVALIDAS"
        dataset_path = r"/mnt/beegfs/ngvargas/datasetsVALIDAS/"
        _, _, _, input_size = load_dataset(dataset_path, augment = 5)
        with open('features.pkl','wb') as f:
            pickle.dump([input_size], f)
    else:
        with open('features.pkl','rb') as f:
            [input_size] = pickle.load(f)

        train_dict, valid_dict, test_dict = {}, {},{}
        for ray in list_rays:
            with open(ray+'.pkl','rb') as f:
                ray_train_dict, ray_valid_dict, ray_test_dict = pickle.load(f)
            train_dict[ray] = ray_train_dict[ray]
            valid_dict[ray] = ray_valid_dict[ray]
            test_dict[ray] = ray_test_dict[ray]
            
    """
    train_dict.keys() == test_dict.keys()
    len(train_dict.keys())
    """

    #list_rays = list(valid_dict.keys()) #son las keys de valid porque si no quedaron fotos para este set no me vale esa raya
    print("We are working with the following ", len(list_rays), " skates:")
    for k in list_rays:
        print(k)


    ##########################################################################################

    # CREATING SIAMESE NETWORK
    
    print("CREATING SIAMESE NETWORK")
    
    slope = -0.05
    net = NN(input_size = input_size, slope = slope)
    
    
    net.to(device)

    """
    print('Red:', net)
    print(summary(model = net, input_size = torch.Size([input_size])))
    """


    ###########################################################################################

    # TRAINING AND VALIDATING LOOP

    print('\nTRAINING\n')

    loss = nn.BCELoss(reduction='mean')
    optimizador = torch.optim.SGD(params = net.parameters(), lr = lr, momentum = momentum)

    #control variables
    train_iter = 0
    continue_training = True
    best_validacc_iteration = 0
    best_validation_accuracy = 0.0 #metric to be maximized
    train_losses, valid_accuracies = [], []

    for _ in range(total_trainiters):
        if continue_training:
        
            if train_iter - best_validacc_iteration > 1000: #STOP CRITERIA, NOT IMPROVING FOR LONG
                print('Early Stopping at iteration: ', train_iter)
                continue_epoch = False
                break


            train_iter += 1

            dataset_train = myDataSet(train_dict, device, input_size, batch_size, list_rays)
            
            #with torch.cuda.device(device):
            if True:
                
                x = dataset_train.X.to(device)
                y = dataset_train.Y.to(device)
                    
                net.train()
                net.to(device)
                    
                y_pred = net.forward(x)
    
                L = loss(y_pred.squeeze(1), y)
                train_losses.append( float(L) )
                print("Train iter: ", train_iter, ", train mean BCE loss over batch is ", float(L))
                L.backward()
                    
                optimizador.step()
                optimizador.zero_grad()
    
                #VALIDACION
                if train_iter % validate_each == 0:
    
                    valid_acc, _ = evaluating('validation', valid_dict, train_dict, list_rays, device, net)
                    valid_accuracies.append(valid_acc)
                    print('Iteration ' , train_iter, '. Validation accuracy: ', valid_acc)
    
                    if ( valid_acc > best_validation_accuracy ):
                            
                        best_validation_accuracy = valid_acc
                        best_validacc_iteration = train_iter
                        torch.save(net.state_dict(), os.getcwd() + '/' + model_name + '.pt')

    print('\nBest validation accuracy: ', best_validation_accuracy, ', at iteration ', best_validacc_iteration) 

    ###########################################################################################

    # TEST
    print('\n TESTING \n')

    test_acc, dict_preds = evaluating('test', test_dict, train_dict, list_rays, device, net, model_name)                           
                                    
    return test_acc
