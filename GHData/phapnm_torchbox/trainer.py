import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train_one_epoch(
        epoch, num_epoch,
        model, device,
        train_loader, valid_loader,
        criterion, optimizer,
        train_metrics, valid_metrics,
    ):
    # train only classsify layer few first epoch and then train all DNN
    if (epoch + 1)  == 5:
        print("Release Parameter")
        for param in model.parameters():
            param.requires_grad = True
    print(('\n' + '%13s' * 3) % ('Epoch', 'gpu_mem', 'mean_loss'))
    # training-the-model
    with tqdm(enumerate(train_loader), total = len(train_loader)) as pbar:
        train_loss = 0
        valid_loss = 0
        mloss = 0
        correct = 0
        model.train()
        for batch_i, (inputs, targets) in pbar:
            # move-tensors-to-GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward model
            outputs = model(inputs)
            # calculate-the-batch-loss
            loss = criterion(outputs, targets)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * inputs.size(0)
            ## calculate training metrics
            outputs_softmax = torch.softmax(outputs, dim=-1)
            probs, preds = torch.max(outputs_softmax.data, dim=-1)
            train_metrics.step(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
            correct += torch.sum(preds.data == targets.data).item()

            ## pbar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            mloss = (mloss * batch_i + loss.item())/(batch_i + 1)
            s = ('%13s' * 2 + '%13.4g' * 1) % ('%g/%g' % (epoch+1, num_epoch), mem, mloss)
            pbar.set_description(s)
            pbar.set_postfix(Lr = optimizer.param_groups[0]['lr'])
        
        train_loss = train_loss/len(train_loader.dataset)
        train_acc = correct/len(train_loader.dataset)
        
    
    #validate-the-model
    with tqdm(enumerate(valid_loader), total = len(valid_loader)) as pbar:
        pbar.set_description(('%26s'  + '%13s'* 3) % ('Train Acc', 'Train Loss', 'Val Acc', 'Val Loss'))
        model.eval()
        all_labels = []
        all_preds = []
        for batch_i, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # update-validation-loss
                valid_loss += loss.item() * inputs.size(0)
                ## calculate training metrics
                outputs_softmax = torch.softmax(outputs, dim=-1)
                
            probs, preds = torch.max(outputs_softmax.data, dim=-1)
            all_labels.extend(targets.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())

        valid_metrics.step(all_labels, all_preds)
        valid_loss = valid_loss/len(valid_loader.dataset)
        valid_acc = accuracy_score(all_labels, all_preds)
        print(('%26.4g' + '%13.4g'* 3) % (train_acc, train_loss ,valid_acc, valid_loss))

    return (
        train_loss, train_acc,
        valid_loss, valid_acc,
        train_metrics.epoch(),
        valid_metrics.last_step_metrics(),
    )
