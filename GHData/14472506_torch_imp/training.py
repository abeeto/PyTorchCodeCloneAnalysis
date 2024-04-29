from ctypes import util
from models import MaskRCNN_model, MaskRCNN_mobilenetv2
from data_loader import COCOLoader
import torch
import utils
from engine import training_loop
import json

def data_loader_config(dir, batch_size):
    """
    funttion task: to configure the data loader using only one string to reduce inputs in the 
                   config dictionary. the function makes the assumption that json is titled
                   the same as the file is it located in. i.e "train".

    inputs: (dir[str]) - A string used to get the json string and to point the COCO loader at 
                         the directory where the data is stored
            
            (batch_size[int]) - passed to the data loader function
    
    outputs: returns a dataload that parses the coco dataset pointed to by the dir
    
    dependancies: - COCOLoader function from data_loader.py
    """
    
    # configuring json string
    json = "/" + dir.split("/")[-1] + ".json"
    
    # loading dataset
    dataset = COCOLoader(dir, dir + json)

    # configuring data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # returing data loader
    return(data_loader)

def main(conf_dict):
    """
    fucntion task: The main function used to execute the training of the network. the function 
                   saves both .pth files of the models and .json files to the output directory
                   specified in the config dict
    
    inputs: (conf_dict[dict]) - The dictionary contains all input parameters which are used 
                                throughout the training process. required inputs can be found
                                at the bottom of the script where they are defined. once a
                                more rigid structure for these have been defined more detials 
                                will be provided
    
    outputs: (best_train_model) - the epoch, model_state_dict and optimiser state dict for the
                                  model the achieved the best training loss whilst training
             
             (best_val_model) - the epoch, model_state_dict and optimiser for the best training
                                loss value whilst the validation loss is value is the best 
            
             (completion message) - The final action of the loop is to print a message to the 
                                    command line alerting the use that training is complete
    
    dependancies: - fix_seed from utils.py
                  - data_loader_config function
                  - Model function from models.py
                  - training loop function from training.py
    """

    # This line should be ran first to ensure a gpu is being used if possible
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # fixing the random seed: 42 is the key!
    utils.fix_seed(42)
    
    # retieving data loaders
    if conf_dict["train_ds"] is not "":
        train_data_loader = data_loader_config(conf_dict["train_ds"], conf_dict['batch_size'])
    if conf_dict["val_ds"] is not "": 
        val_data_loader = data_loader_config(conf_dict["val_ds"], conf_dict['batch_size'])
    if conf_dict["test_ds"] is not "":
        test_data_loader = data_loader_config(conf_dict["test_ds"], conf_dict['batch_size'])

    # get the model from model function and load it to device
    #model = MaskRCNN_model(conf_dict['num_classes'])  
    model = MaskRCNN_mobilenetv2(conf_dict['num_classes'])
    model.to(device)  

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    start_epoch = 0
    
    # loading model data if specidied
    if conf_dict["load"] is not "":
        checkpoint = torch.load(conf_dict["load"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    
    # learning rate scheduler ! implement LR scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                               step_size=3,
    #                                               gamma=0.1)

    training_loop(model, device, optimizer, train_data_loader, val_data_loader, start_epoch,
                  conf_dict["num_epochs"], conf_dict["print_freq"], conf_dict["out_dir"],
                  conf_dict["val_freq"])

    print("training complete")

if __name__ == "__main__":
    
    # defining configurabels
    conf_dict = {}
    conf_dict["train_ds"] = "data/jersey_royal_dataset/train"
    conf_dict["val_ds"] = "data/jersey_royal_dataset/val"
    conf_dict["test_ds"] = "data/jersey_royal_dataset/test"

    conf_dict["batch_size"] = 1
    conf_dict["num_classes"] = 2 
    conf_dict["num_epochs"] = 20
    conf_dict["print_freq"] = 20
    conf_dict["val_freq"] = 5   
    
    conf_dict["out_dir"] = "output/Mask_RCNN_fix_test5"
    conf_dict["load"] = ""

    # call main
    main(conf_dict)