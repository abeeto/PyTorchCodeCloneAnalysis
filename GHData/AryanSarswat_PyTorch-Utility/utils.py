import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import numpy as np
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt

def sample_from_data(dataset, sample_size = 4, show_gray = False):
    """Function to sample from dataset

    Args:
        dataset (_type_): _description_
        sample_size (int, optional): _description_. Defaults to 4.
        show_gray (bool, optional): _description_. Defaults to False.
    """
    pass

def modelSummary(model, verbose=False):
    if verbose:
        print(model)
    
    total_parameters = 0
        
    for name, param in model.named_parameters():
        num_params = param.size()[0]
        total_parameters += num_params
        if verbose:
            print(f"Layer: {name}")
            print(f"\tNumber of parameters: {num_params}")
            print(f"\tShape: {param.shape}")
    
    if total_parameters > 1e5:
        print(f"Total number of parameters: {total_parameters/1e6:.2f}M")
    else:
        print(f"Total number of parameters: {total_parameters/1e3:.2f}K") 

def train_epoch(model: nn.Module, device: torch.device, train_dataloader: DataLoader, training_params: dict, metrics: dict):
    """_summary_

    Args:
        model (nn.Module): Model to be trained by
        device (str): device to be trained on
        train_dataloader (nn.data.DataLoader): Dataloader object to load batches of dataset
        training_params (dict): Dictionary of training parameters containing "batch_size", "loss_function"
                                "optimizer".
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        run_results (dict): Dictionary of metrics computed for the epoch
    """
    BATCH_SIZE = training_params["batch_size"]
    LOSS_FUNCTION = training_params["loss_function"]
    OPTIMIZER = training_params["optimizer"]
    
    model = model.to(device)
    model.train()
    
    # Dictionary holding result of this epoch
    run_results = dict()
    for metric in metrics:
        run_results[metric] = 0.0
    run_results["loss"] = 0.0
    
    # Iterate over batches
    num_batches = 0
    for x, target in train_dataloader:
        num_batches += 1

        # Move tensors to device
        input = x.to(device)
        
        # Forward pass
        output = model(input)
        
        # Compute loss
        loss = LOSS_FUNCTION(output, target)
        
        # Backward pass
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        # Update metrics
        run_results["loss"] += loss.detach().item()
        for key, func in metrics.items():
            run_results[key] += func(output, target).detach().item()
            
        # Clean up memory
        del loss
        del input
        del output
        
    for key in run_results:
        run_results[key] /= num_batches
    
    return run_results

def evaluate_epoch(model: nn.Module, device: torch.device, validation_dataloader: DataLoader, training_params: dict, metrics: dict):
    """_summary_

    Args:
        model (nn.Module): model to evaluate
        device (str): device to evaluate on
        validation_dataloader (DataLoader): DataLoader for evaluation
        training_params (dict): Dictionary of training parameters containing "batch_size", "loss_function"
                                "optimizer".
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        run_results (dict): Dictionary of metrics computed for the epoch
    """
    LOSS_FUNCTION = training_params["loss_function"]
    
    model = model.to(device)
    
    # Dictionary holding result of this epoch
    run_results = dict()
    for metric in metrics:
        run_results[metric] = 0.0
    run_results["loss"] = 0.0
    
    # Iterate over batches
    with torch.no_grad():
        model.eval()
        num_batches = 0
        
        for x, target in validation_dataloader:
            num_batches += 1
            
            # Move tensors to device
            input = x.to(device)
            
            # Forward pass
            output = model(input)
            
            # Compute loss
            loss = LOSS_FUNCTION(output, target)
            
            # Update metrics
            run_results["loss"] += loss.detach().item()
            for key, func in metrics.items():
                run_results[key] += func(output, target).detach().item()
                
            # Clean up memory
            del loss
            del input
            del output
                
    for key in run_results:
        run_results[key] /= num_batches
        
    return run_results

def train_evaluate(model: nn.Module, device: torch.device, train_dataset: Dataset, validation_dataset: Dataset, training_params: dict, metrics: dict):
    """Function to train a model and provide statistics during training

    Args:
        model (nn.Module): Model to be trained
        device (torch.device): Device to be trained on
        train_dataset (DataLoader): Dataset to be trained on
        validation_dataset (DataLoader): Dataset to be evaluated on
        training_params (dict): Dictionary of training parameters containing "num_epochs", "batch_size", "loss_function",
                                                                             "save_path", "optimizer"
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        _type_: _description_
    """
    NUM_EPOCHS = training_params["num_epochs"]
    BATCH_SIZE = training_params["batch_size"]
    SAVE_PATH = training_params["save_path"]
    SAMPLE_SIZE = 10
    PLOT_EVERY = 1
    
    # Initialize metrics
    train_results = dict()
    train_results['loss'] = []
    evaluation_results = dict()
    evaluation_results['loss'] = []
    
    for metric in metrics:
        train_results[metric] = []
        evaluation_results[metric] = []
    
    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        print(f"======== Epoch {epoch+1}/{NUM_EPOCHS} ========")

        # Train Model
        print("Training ... ")
        epoch_train_results = train_epoch(model, device, train_dataloader, training_params, metrics)
        

        # Evaluate Model
        print("Evaluating ... ")
        epoch_evaluation_results = evaluate_epoch(model, device, validation_dataloader, training_params, metrics)
        
        for metric in metrics:
            train_results[metric].append(epoch_train_results[metric])
            evaluation_results[metric].append(epoch_evaluation_results[metric])
            
        
        # Print results of epoch
        print(f"Completed Epoch {epoch+1}/{NUM_EPOCHS} in {(time.time() - start):.2f}s")
        print(f"Train Loss: {epoch_train_results['loss']:.4f} \t Validation Loss: {epoch_evaluation_results['loss']:.4f}")
        
        # # Plot results
        # if epoch % PLOT_EVERY = 0:
        #     batch = next(iter(validation_dataloader))
            
        #     model.eval()
        #     ouputs = model(batch[0].to(device)).detach().cpu()
            
        #     fig, ax = plt.subplots(2, SAMPLE_SIZE, figsize=(SAMPLE_SIZE * 5,15))
        #     for i in range(SAMPLE_SIZE):
        #         image = batch[0][i].detach().cpu()
        #         output = ouputs[i]
                
        #         ax[0][i].imshow(image.reshape(28,28))
        #         ax[1][i].imshow(output.reshape(28,28))
            
        #     plt.savefig(f"{SAVE_PATH}_epoch{epoch + 1}.png")
        #     plt.close()
        
        # # Save model
        # SAVE = f"{SAVE_PATH}_epoch{epoch + 1}.pt"
        # torch.save(model.state_dict(), SAVE)
           
    return train_results, evaluation_results

def plot_training_results(train_results, validation_results):
    """Function to plot training results

    Args:
        train_results (dict): Dictionary of training results
        validation_results (dict): Dictionary of validation results
    """
    plt.plot(train_results['loss'], label='Training Loss')
    plt.plot(train_results['l1'], label='Training L1 Loss')
    plt.plot(train_results['l1_norm'], label='Training L1 Norm')
    plt.plot(validation_results['loss'], label='Validation Loss')
    plt.plot(validation_results['l1'], label='Validation L1 Loss')
    plt.plot(validation_results['l1_norm'], label='Validation L1 Norm')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    
class BasicNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicNet,self).__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.output(x)
        return x
    
if __name__ == '__main__':
    input_size = 512
    hidden_dimension = 5096
    output_size = 16
    
    model = BasicNet(input_size, hidden_dimension, output_size)
    
    modelSummary(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    training_data_size = 2**12
    validation_data_size = int(training_data_size * 0.2)
    
    training_data = torch.rand(training_data_size, input_size)
    training_target = torch.rand(training_data_size, output_size)
    
    validation_data = torch.rand(validation_data_size, input_size)
    validation_target = torch.rand(validation_data_size, output_size)
    
    train_dataset = TensorDataset(training_data, training_target)
    validation_dataset = TensorDataset(validation_data, validation_target)
    
    training_params = {
        'num_epochs': 50,
        'batch_size': 512,
        'loss_function':F.mse_loss,
        'optimizer': torch.optim.Adam(model.parameters(), lr=0.001),
        'save_path': './model.pt'
    }
    
    metrics = {
        'l1': lambda output, target: torch.mean(torch.abs(output - target)),
        'l1_norm': lambda output, target: torch.norm(output - target, 1),
    }
    
    train_results, evaluation_results = train_evaluate(model, device, train_dataset, validation_dataset, training_params, metrics)
    plot_training_results(train_results=train_results, validation_results=evaluation_results)