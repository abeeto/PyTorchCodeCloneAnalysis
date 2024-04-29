import torch
from tqdm import tqdm

def test(dataloader,model,criterion,device = "cpu"):
    losses = []
    loader = tqdm(dataloader)
    correct = 0
    nums = 0
    model.eval()
    with torch.no_grad():
        for data,target in loader:
            data,target = data.to(device),target.to(device)
            pred = model(data)
            loss = criterion(pred,target)
            pred_label = pred.max(1)[1]
            correct += (pred_label == target).sum()
            nums += data.shape[0]
            losses.append(loss.item())
            loader.set_postfix(test_loss_batch = loss.item())
    
    model.train()
    return sum(losses)/len(losses),correct/nums*100