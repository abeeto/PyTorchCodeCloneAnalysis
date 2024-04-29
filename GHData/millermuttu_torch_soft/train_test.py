from tqdm import tqdm
import torch
import numpy as np


def train(model, dataloader, optimizer, loss_fn, device, train_losses, train_acc):
    """ traing the model with dataloader and specified optimizer and loss function"""
    model.train()
    pbar = tqdm(dataloader, total=len(dataloader))
    correct = 0
    processed = 0
    train_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdims=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_epoch_loss += loss.item()
        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Accuracy={(100 * correct / processed):.2f}'
        )

    train_losses.append(train_epoch_loss / len(dataloader.dataset))
    train_acc.append(100. * correct / processed)

    # return model, train_losses, train_acc


def test(model, dataloader, loss_fn, device, n_misclassified, test_losses, test_accuracy, misclassified_imgs):
    model.eval()
    # pbar = tqdm(dataloader, total=len(dataloader))
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            batch_preds = model(data)
            test_loss += loss_fn(batch_preds, target).item()
            preds = batch_preds.argmax(dim=1, keepdims=True)
            correct += preds.eq(target.view_as(preds)).sum().item()

            if len(misclassified_imgs) < n_misclassified:
                incorrect_id = ~preds.eq(target.view_as(preds))
                incorrect_id = incorrect_id.cpu().numpy().ravel()
                incorrect_id = np.where(incorrect_id == True)[0]
                if len(incorrect_id) != 0:
                    for i in incorrect_id:
                        if len(misclassified_imgs) < n_misclassified:
                            misclassified_imgs.append({'img': data[i],
                                                       'pred': preds[i].item(),
                                                       'target': target.view_as(preds)[i].item()})
                        else:
                            break

    test_loss /= len(dataloader.dataset)
    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / len(dataloader.dataset))

    print(
        f'\nValidation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({test_accuracy[-1]:.2f}%)\n'
    )

    return test_loss

def train_ocp(model, dataloader, optimizer,sheduler,loss_fn, device, train_losses, train_acc):
    """ traing the model with dataloader and specified optimizer and loss function"""
    model.train()
    pbar = tqdm(dataloader, total=len(dataloader))
    correct = 0
    processed = 0
    train_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdims=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_epoch_loss += loss.item()
        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Accuracy={(100 * correct / processed):.2f}'
        )
        sheduler.step()

    train_losses.append(train_epoch_loss / len(dataloader.dataset))
    train_acc.append(100. * correct / processed)