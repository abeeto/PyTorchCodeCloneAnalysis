import torch
from tqdm import tqdm

def train(dataloader,model,criterion,optimizer,device="cpu",isEval=False):
    losses = []
    loader = tqdm(dataloader)
    if isEval:
        nums = 0
        correct = 0
        
    for data,target in loader:
        optimizer.zero_grad()
        data,target = data.to(device),target.to(device)
        pred = model(data)
        loss = criterion(pred,target)
        loss.backward()
        optimizer.step()
        
        
        if isEval:
            nums += data.shape[0]
            pred_label = pred.max(1)[1]
            correct += (pred_label==target).sum()
        
        loader.set_postfix(train_loss_batch = loss.item())
        losses.append(loss.item())
    if isEval:
        return sum(losses)/len(losses),correct/nums*100

    else:
        return sum(losses)/len(losses)

    