import torch
import pandas as pd
from tqdm import tqdm

def train(model, train_loader, criterion_cls, optimizer, device, writer, cur_epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, ncols=100,
                desc='Epoch {}'.format(cur_epoch))
    offset = (cur_epoch-1)*len(train_loader)
    for idx, (data, gt) in pbar:
        data, gt = data.to(device), gt.to(device)

        out = model(data)
        loss = criterion_cls(out, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss':'{:.5f}'.format(loss.item())})
        writer.add_scalar('train/loss', loss.item(), offset + idx)

def validate(model, val_loader, device, writer, cur_epoch):
    model.eval()
    total_cnt = 0
    total_correct_cnt = 0
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), leave=False, ncols=100,
                desc='Val {}'.format(cur_epoch))
    with torch.no_grad():
        for idx, (data, gt) in pbar:
            data, gt = data.to(device), gt.to(device)

            out = model(data)
            _, pred = torch.max(out, dim=1)

            total_cnt += pred.size(0).item()
            total_correct_cnt += (pred == gt).sum().item()

    val_acc = total_correct_cnt / total_cnt

    print('Epoch {} | Val Acc: {:.5f}'.format(cur_epoch, val_acc))
    writer.add_scalar('val/acc', val_acc, cur_epoch)

    return val_acc

def test(model, test_loader, device, out_path):
    pred_list = []
    name_list = []

    model.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False, ncols=100, desc='Testing')

    with torch.no_grad():
        for idx, (data, name) in pbar:
            if idx == 0:
                assert data.size(0) == 1
            data = data.to(device)

            out = model(data)
            _, pred = torch.max(out, dim=1)

            pred_list.append(pred.item())
            name_list.append(name)
    
    out_df = pd.DataFrame({'name': name_list, 'pred': pred_list})
    out_df.to_csv(out_path, index=False)