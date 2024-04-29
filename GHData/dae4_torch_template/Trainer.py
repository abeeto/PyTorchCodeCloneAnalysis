import torch
from tqdm import tqdm
import os
import shutil

def save_checkpoint(state, epoch, val_acc, is_best, checkpoint):
    filepath = os.path.join(checkpoint, f"food_weight-{epoch:02d}-{val_acc:.2f}.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint, exist_ok=True)
    
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))

def val(val_loader,model, criterion, device,epoch,max_epoch):
    model.eval()

    performance_dict = {
        "epoch": epoch+1
    }

    summ = {
        "val_loss": 0,
        "val_acc": 0
    }
    with torch.no_grad():
        with tqdm(val_loader) as t:
            t.set_description(f'[{epoch+1}/{max_epoch}]')
            for i,data in enumerate(t):
                val_batch, labels_batch = data
                val_batch = val_batch.to(device)
                labels_batch = labels_batch.to(device) 
                # 변화도(Gradient) 매개변수를 0으로 만들고

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = model(val_batch)
                loss = criterion(outputs, labels_batch)

                # 통계를 출력합니다.
                summ["val_loss"] += loss.item()

                batch_size = labels_batch.size(0)
                _, preds = torch.max(outputs.data, 1)
                correct = (preds == labels_batch).sum().item()
                summ["val_acc"] += correct / batch_size

                dict_summ = {"val_loss" : f'{summ["val_loss"]/(i+1):05.3f}'}
                dict_summ.update({"val_acc" : f'{summ["val_acc"]/(i+1)*100:05.3f}'})
                t.set_postfix(dict_summ)
                t.update()

    performance_dict.update({key : val/(i+1) for key, val in summ.items()})
    return performance_dict

def train(train_loader,model,device,optimizer,criterion,epoch,max_epoch):

    model.train()

    performance_dict = {
        "epoch": epoch+1
    }


    summ = {
            "loss": 0,
            "acc": 0
        }
    with tqdm(train_loader) as t:
        t.set_description(f'[{epoch+1}/{max_epoch}]')
        for i,data in enumerate(t) :
            train_batch, labels_batch = data
            train_batch = train_batch.to(device)
            labels_batch = labels_batch.to(device) 
            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(train_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            summ["loss"] += loss.item()

            batch_size = labels_batch.size(0)
            _, preds = torch.max(outputs.data, 1)
            correct = (preds == labels_batch).sum().item()
            summ["acc"] += correct / batch_size

            dict_summ = {"loss" : f'{summ["loss"]/(i+1):05.3f}'}
            dict_summ.update({"acc" : f'{summ["acc"]*100/(i+1):05.3f}'})
            t.set_postfix(dict_summ)
            t.update()

    performance_dict.update({key: val/(i+1) for key, val in summ.items()})

    return performance_dict