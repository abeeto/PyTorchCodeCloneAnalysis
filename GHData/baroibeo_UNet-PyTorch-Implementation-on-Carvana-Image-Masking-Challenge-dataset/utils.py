import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transform(resized_height,resized_width,train=True):
    if train:
        return A.Compose([
                    A.Resize(height=resized_height,width=resized_width),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.Normalize(
                        mean=[0.0,0.0,0.0],
                        std=[1.0,1.0,1.0],
                        max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ])
    else:
        return A.Compose([
                    A.Resize(height=resized_height,width=resized_width),
                    A.Normalize(
                        mean=[0.0,0.0,0.0],
                        std=[1.0,1.0,1.0],
                        max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ])

def train_fn(model,dataloader,loss_fn,optimizer,device="cpu"):
    model=model.to(device)
    loader=tqdm(dataloader)
    losses=[]
    for data,target in loader:
        data=data.to(device)
        target=target.to(device)
        target=target.float().unsqueeze(1)
        pred=model(data)
        
        loss=loss_fn(pred,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loader.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
      

def test_fn(model,dataloader,loss_fn,device="cpu"):
    model.eval()
    num_corrects=0
    num_pixels=0
    dice_scores=[]
    with torch.no_grad():
        losses=[]
        for data,target in dataloader:
            data=data.to(device)
            target=target.to(device)
            target=target.float().unsqueeze(1)
            preds=model(data)
            loss=loss_fn(preds,target).item()
            losses.append(loss)
            pred=torch.sigmoid(preds)
            pred=(pred>0.5).float()
            num_corrects+=(pred==target).sum()
            num_pixels+=torch.numel(pred)
            dice_scores.append((2*(pred*target).sum())/((pred+target).sum()+1e-8))
            
    
    model.train()
    return sum(losses)/len(losses),num_corrects/num_pixels*100,sum(dice_scores)/len(dice_scores)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])