from typing import Dict
from tqdm import tqdm
import torch


def eval(model, valid_loader, criterion, device) -> Dict[str, float]:
    results = dict()
    model.eval()
    correct, total, total_loss = 0., 0., 0.

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (pred == labels).sum().item()

    results['val_acc'] = correct / total
    results['val_loss'] = total_loss / total
    return results


def train_a_epoch(model, train_loader, criterion,
                  optimizer, device) -> Dict[str, float]:
    results = dict()
    model.train()
    train_loss, train_num = 0., 0.

    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_num += labels.size(0)
        train_loss += loss.item()
    train_loss /= train_num

    results['train_loss'] = train_loss
    return results


def train(epochs, model, train_loader, valid_loader,
          criterion, optimizer, scheduler, device):
    result = dict()
    for ep in range(epochs):
        scheduler.step()

        # train
        results_train = train_a_epoch(model=model,
                                      train_loader=train_loader,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      device=device)
        train_loss = results_train['train_loss']

        # valid
        results_eval = eval(model=model, valid_loader=valid_loader,
                            device=device, criterion=criterion)
        val_acc = results_eval['val_acc']
        val_loss = results_eval['val_loss']

        # log
        result["train_loss"] = train_loss
        result["val_loss"] = val_loss
        result["val_acc"] = val_acc
        print(f"Epoch {ep+1} loss = {train_loss:.06} val_acc = {val_acc:.04}")
    return result
