from __future__ import print_function
import os
import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm
import random
import numpy as np

def make_dataloader(path:str, batch_size:int, n_train_ratio=80):
    """return dataloader for train and validation.

    Args:
        path (str): dataset root path.
        batch_size (int): number of batch of dataloader.
    
    Returns:
        train_loader: Dataloader for train.
        val_loader: Dataloader for validation.
    """

    print('Preparing dataloader...')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(path, transform)
    n_train = int(len(dataset) * n_train_ratio / 100)
    n_val = int(len(dataset) - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader


def compose_net(model_cfg: OmegaConf):
    """Automatically compose PyTorch Model from configs.

    Args:
        model_cfg (OmegaConf): model configs.
        
    Returns:
        net: PyTorch Model.
    """

    # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True  # <- enable it when running on docker.
    MODEL_LIST = [
        torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=False),
        torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_efficientnet_b0', pretrained=False),
        torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_efficientnet_b4', pretrained=False),
        torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_efficientnet_widese_b4', pretrained=False),
    ]
    
    OUTPUT_SIZE = 26
    
    net = MODEL_LIST[model_cfg.id]
    if model_cfg.id >= 0 and model_cfg.id <= 1:
        net.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, OUTPUT_SIZE),
        )
    elif model_cfg.id >= 2 and model_cfg.id <= 6:
        net.fc = torch.nn.Linear(net.fc.in_features, OUTPUT_SIZE)
    elif model_cfg.id >= 7 and model_cfg.id <= 9:
        net.classifier.fc = torch.nn.Linear(net.classifier.fc.in_features, OUTPUT_SIZE)
    print(net)
    
    return net


def train_loop(device: str, run_id: int,
        net, 
        optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        _max_epoch=100, 
        _eary_stop=10
    ) -> float:
    """iterate train and evaluation.

    Args:
        device (str): device information. cuda or cpu.
        run_id (int): automatically allocated by main().
        net         : PyTorch Model
        optimizer   : PyTorch Optimizer
        criterion   : PyTorch Criterion
        train_loader: DataLoader for train
        val_loader  : DataLoader for validation
        _max_epoch  : max train loop counts
        _eary_stop  : eary stopping
    
    Returns:
        float: return best evaluation accuracy. 
    """

    best_acc = eary_stop = 0
    for epoch in range(_max_epoch):
        print('\nEpoch: %d' % epoch)
        t_loss, t_acc, v_loss, v_acc = 0, 0, 0, 0
        
        # Train Phase. ---------------------------------------------------
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                t_loss = train_loss/(batch_idx+1)
                t_acc = 100.*correct/total   
                pbar.set_postfix(loss=t_loss, acc=t_acc)
                pbar.update(1)
                
        mlflow.log_metric(f'train loss v_{run_id:02d}', t_loss, step=epoch)
        mlflow.log_metric(f'train acc v_{run_id:02d}', t_acc, step=epoch)
        #---------------------------------------------------

        # Validation Phase. ---------------------------------------------------
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with tqdm(total=len(val_loader)) as pbar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    v_loss = test_loss/(batch_idx+1)
                    v_acc = 100.*correct/total
                    pbar.set_postfix(loss=v_loss, acc=v_acc)
                    pbar.update(1)
                    
        mlflow.log_metric(f'val loss v_{run_id:02d}', v_loss, step=epoch)
        mlflow.log_metric(f'val acc v_{run_id:02d}', v_acc, step=epoch)
        #---------------------------------------------------

        # Save best checkpoint. ---------------------------------------------------
        if v_acc > best_acc:
            print('Saving...')
            state = {
                'net': net.state_dict(),
                'acc': v_acc,
                'epoch': epoch,
                'optimizer': optimizer
            }

            torch.save(state, f'best-ckpt-{run_id:02d}.t7')
            mlflow.log_artifact(f'best-ckpt-{run_id:02d}.t7')
            best_acc = v_acc
            eary_stop = 0
        else:
            eary_stop += 1
            
        if eary_stop >= _eary_stop:
            break
        #---------------------------------------------------
    
    # Save last checkpoint. ---------------------------------------------------
    print('Saving...')
    state = {
        'net': net.state_dict(),
        'acc': v_acc,
        'epoch': epoch,
        'optimizer': optimizer
    }

    torch.save(state, f'last-ckpt-{run_id:02d}.t7')
    mlflow.log_artifact(f'last-ckpt-{run_id:02d}.t7')
    #---------------------------------------------------
    
    return best_acc


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig) -> float:
    """main training function.

    Args:
        cfg (DictConfig): configs from hydra and optuna. 

    Returns:
        float: return acc of trained model.
    """
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True              # True: Improves training speed on GPU.
    torch.backends.cuda.matmul.allow_tf32 = False      # False: Improves numerical accuracy on GPU.
    torch.backends.cudnn.allow_tf32 = False            # False: Improves numerical accuracy on GPU.
    
    
    ## mlflow settings. ---------------------------------------------------
    mlflow.set_tracking_uri(f'file://{hydra.utils.get_original_cwd()}/mlruns')
    mlflow.set_experiment(cfg.mlflow.exp_name)
    
    with mlflow.start_run():
        mlflow.log_params(cfg.model)
        mlflow.log_params(cfg.optimizer)

        ## start mlflow trace. ---------------------------------------------------
        acc = np.zeros((cfg.num_trials))
        for run_id in range(cfg.num_trials):
            print(f'\nStart model training loops [{(run_id+1):02d}/{cfg.num_trials:02d}]...')
            print('Preparing a Network...')
            net = compose_net(cfg.model)
            net = net.to(device)

            print('Setting a optimizer and a criterion... ')
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), betas=cfg.optimizer.betas, lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
            
            
            print(f'Training loops...')
            train_loader, val_loader = make_dataloader(cfg.data, cfg.batch_size)
            acc[run_id] = train_loop(device, run_id, net, optimizer, criterion, train_loader, val_loader, cfg.epochs, cfg.eary_stop)
        
        mlflow.log_metric('total score', np.mean(acc))
        #---------------------------------------------------

        # Save hydra parameters. ---------------------------------------------------
        mlflow.log_artifact('.hydra/config.yaml')
        mlflow.log_artifact('.hydra/hydra.yaml')
        mlflow.log_artifact('.hydra/overrides.yaml')
        mlflow.log_artifact(os.path.basename(__file__).replace('.py', '.log'))
        #---------------------------------------------------
    
        mlflow.end_run()
    
    return np.mean(acc)


if __name__ == "__main__":
    main()
