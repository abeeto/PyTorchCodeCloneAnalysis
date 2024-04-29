# training.py

import torch
import torch.nn as nn
from tqdm import tqdm


def train(train_loader, model, optimizer, device):
    model.train()

    for data in tqdm(train_loader, desc='training'):
        inputs = data['image']
        targets = data['targets']

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs = torch.argmax(outputs, dim=1).to(device, dtype=torch.float)
        loss = nn.CrossEntropyLoss()(outputs, targets) # .view(-1, 1))
        loss.backward()
        optimizer.step()


def evaluate(valid_loader, model, device):
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():

        for data in tqdm(valid_loader, desc='evaluating'):
            inputs = data['image']
            targets = data['targets']

            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()

            final_targets.extend(targets)
            final_outputs.extend(outputs)

    return final_outputs, final_targets
