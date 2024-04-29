import torch
import config
from data_loader import *

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def dataset_loader():

    train_dataset = BrainLoader(config.TRAIN_PATH,
                                train=True)

    test_dataset = BrainLoader(config.TEST_PATH,
                               train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )
    return train_loader, test_loader

def iou_calc(pred, target):
    intersection = np.sum(np.abs(pred * target))
    union = (np.sum(pred) + np.sum(target)) - intersection
    return np.mean((intersection + 1) / (union + 1))