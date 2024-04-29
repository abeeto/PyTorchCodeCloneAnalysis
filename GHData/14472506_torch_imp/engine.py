###### imports ####################################################################################

import utils
from matplotlib import pyplot as plt
import torch
import numpy as np
import math
import sys
import json



##### functions ###################################################################################

#def check_trigger(loss_val, results_dict):
#    r = results_dict['val_loss']
#    try:
#        eval = np.array([r[-3], r[-2], r[-1]], dtype=float)
#        #grad = np.gradient(eval, 2)
#        print(eval)
#    except IndexError:
#        pass



def checkpoint(epoch, model, optimizer, out_dir, file_name):
    """
    name: checkpoint

    function: The function handles the process for saving a the model

    inputs:

    outputs:
    """
    # generating state checkpoint dict
    utils.mkdir(out_dir + "/checkpoints/")
    
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    # file path            
    file_path = out_dir + "/checkpoints/" + file_name

    # saving checkpoint
    torch.save(checkpoint, file_path)



def save_models(epoch, optimizer, model, results_dict, out_dir, loss_value, iter_count):
    """
    name: save_models

    function: The function handles the logic behind saving models in the training loop.
              The function makes use of the save checkpoint function where appropriate

    inputs:

    outputs:
    """
    # load meta data for saving model
    file = out_dir + "/models_metadata.json"
    try:
        metadata = open(file)
        data = json.load(metadata)
    except FileNotFoundError:
        data = {
            "min_train_val": None,
            "min_train_iter": None,
            "min_val_val": None,
            "min_val_iter": None,
            "min_val_train_loss": None
        }

    # saving best train model
    if not data['min_train_val'] or loss_value < data['min_train_val']:
        file_name = "best_train_model.pth" 
        checkpoint(epoch, model, optimizer, out_dir, file_name)
        data['min_train_val'] = loss_value
        data['min_train_iter'] = iter_count

    try:
    # saving best val model
    # first statement checks to see if min val exists or val_loss is less then equal to min_val_loss
        if not data['min_val_val'] or data['min_val_val'] >= min(results_dict['val_loss']):

            # if val_loss is smaller, reset val_train_loss_val
            if not data['min_val_val'] or data['min_val_val'] > min(results_dict['val_loss']):
                data['min_val_train_loss'] = 100 # random large number

                # if smallest train loss for min val loss
                if not data['min_val_train_loss'] or loss_value < data['min_val_train_loss']: 
                    file_name = "best_val_model.pth" 
                    checkpoint(epoch, model, optimizer, out_dir, file_name)
                    data['min_val_val'] = min(results_dict['val_loss'])
                    data['min_val_train_loss'] = loss_value
                    data['min_val_iter'] = iter_count
    except ValueError:
        pass

    # save metadata for models
    with open(file, 'w') as f:
        json.dump(data, f)



def train_one_step(images, targets, model, device, optimizer, results_dict, out_dir, epoch,
                   iter_count, metric_logger):
    """
    name: training_one_step

    function: The function essentially carrys out a single step of the training loop

    inputs:

    outputs:
    """
    # loading images to gpu
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # getting loss data from model and processing data for logging
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())  
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())         
    loss_value = losses_reduced.item()

    # if not zero/infinate
    if not math.isfinite(loss_value):
        print(f"Loss is {loss_value}, stopping training")
        print(loss_dict_reduced)
        sys.exit(1)
    
    ## saving model best models
    save_models(epoch, optimizer, model, results_dict, out_dir, loss_value, iter_count)
    #if not results_dict['train_loss'] or loss_value < min(results_dict['train_loss']):
    #    file_name = "best_train_model.pth" 
    #    checkpoint(epoch, model, optimizer, out_dir, file_name)
    #if not results_dict['val_loss'] or loss

    # appending loss value to lost list

    results_dict['train_loss'].append(loss_value)
    results_dict['train_epoch'].append(iter_count)

    # zero the gradient stored gadient before carrying out
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return results_dict



def val_one_step(images, targets, model, device, loss_list):
    """
    name: val_one_step

    function: The function carrys out a single step of the validation process

    inputs:

    outputs:
    """
    # loading images to gpu
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # getting loss data from model and processing data for logging
    loss_dict = model(images, targets) 
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())          
    loss_value = losses_reduced.item()
    
    # if not zero/infinate
    if not math.isfinite(loss_value):
        print(f"Loss is {loss_value}, stopping training")
        print(loss_dict_reduced)
        sys.exit(1)
    
    loss_list.append(loss_value)
    
    return loss_list



def train_one_epoch(print_freq, train_loader, val_loader, model, device, optimizer, results_dict, out_dir,
                         epoch, val_freq, iter_count):
    """
    name: train_one_epoch

    function: The function loops throught the images in an epoch passing them to the step evaluator
              along with managing tasks at the epoch loop level such as iteration evaluation

    inputs:

    outputs:
    """
    # config model for training
    model.train()

    # initialise logging
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    
    # iterating through images in data loader 
    for images, targets in metric_logger.log_every(train_loader, print_freq, header):
        
        # Eval model here:
        # this must be carried out first as to not be behind the backwards call in the 
        # train one step function.
        val_flag = iter_count/val_freq
        if val_flag % 1 == 0:
            with torch.no_grad():
                val_loss_value = val_one_epoch(model, device, val_loader)
                results_dict['val_loss'].append(val_loss_value)
                results_dict['val_epoch'].append(iter_count)
                model.train()

        # train one step
        results_dict = train_one_step(images, targets, model, device, optimizer, results_dict, 
                                      out_dir, epoch, iter_count, metric_logger)
        
        iter_count += 1
    
    return results_dict, iter_count



def val_one_epoch(model, device, data_loader):
    """
    name: train_one_epoch

    function: The function loops throught the images in an epoch passing them to the step evaluator
              along with managing tasks at the epoch loop level such as iteration evaluation

    inputs:

    outputs:
    """
    # config model for training
    model.train()
    loop_list = []
    
    for images, targets in data_loader:
        loop_list = val_one_step(images, targets, model, device, loop_list)
    
    loss_val = sum(loop_list)/len(loop_list)

    return loss_val



def training_loop(model, device, optimizer, train_data_loader, val_data_loader, 
                    start_epoch, num_epochs, print_freq, out_dir, val_freq):
    """
    name: train_loop

    function: The main function for execution the training loop, this function loops over epochs
              and calls all functions and sub functions, along side this the loop handles data
              recording

    inputs:

    outputs:
    """    
    # initialising data capture
    results_dict = {
        "train_loss": [],
        "train_epoch": [],
        "val_loss": [],
        "val_epoch": []
    }
     
    # epoch counter
    epoch_count = start_epoch + 1

    # TO DO: maybe find a way to integrate epoch into this, or will it append to iter_count?
    iter_count = 1

    for epoch in range(start_epoch, num_epochs):
        results_dict, new_count = train_one_epoch(print_freq, train_data_loader, val_data_loader, 
                                                  model, device, optimizer, results_dict, 
                                                  out_dir, epoch, val_freq, iter_count)
        
        iter_count = new_count
        
        #with torch.no_grad():
        #    val_loss_list = val_one_epoch(model, device, val_loss_list, val_data_loader,
        #                                 epoch, print_freq)

        epoch_count += 1

        # this should work
        file_name = out_dir + "/results.json"
        with open(file_name, 'w') as file:
            json.dump(results_dict, file)