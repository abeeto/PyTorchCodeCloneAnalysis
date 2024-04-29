import torch
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    model.train()

    for data in data_loader:
        # print(data)
        reviews = data['review']
        targets = data['target']
        
        reviews = reviews.to(device, dtype = torch.long)
        targets = targets.to(device, dtype = torch.float)
        
        optimizer.zero_grad()

        predictions = model(reviews)
        
        loss = nn.BCEWithLogitsLoss()
        output = loss(predictions, targets.view(-1,1))
        output.backward()
        optimizer.step()

def evaluate(data_loader, model, device):
    final_predictions = []
    final_targets = []

    model.eval()

    with torch.no_grad():
        for data in data_loader:
            reviews = data['review']
            targets = data['target']
            reviews = reviews.to(device, dtype = torch.long)
            targets = targets.to(device, dtype = torch.float)

            predictions = model(reviews)

            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()

            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return final_predictions, final_targets