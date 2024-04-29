import numpy as np
import time
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from utils import helper as hl
from utils import dataset as dt
from utils import config as cfg 



# ====================================================
# Train loop
# ====================================================

LOGGER = hl.init_logger()

def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = dt.TrainDataset(train_folds,
                                 transform=dt.get_transforms(data='train'))
    valid_dataset = dt.TrainDataset(valid_folds,
                                 transform=dt.get_transforms(data='valid'))

    train_loader = dt.DataLoader(train_dataset,
                              batch_size=cfg.CFG.batch_size,
                              shuffle=True,
                              num_workers=cfg.CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = dt.DataLoader(valid_dataset,
                              batch_size=cfg.CFG.batch_size,
                              shuffle=False,
                              num_workers=cfg.CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if cfg.CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=cfg.CFG.factor, patience=cfg.CFG.patience, verbose=True, eps=cfg.CFG.eps)
        elif cfg.CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, T_max= cfg.CFG.T_max, eta_min=cfg.CFG.min_lr, last_epoch=-1)
        elif cfg.CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.CFG.T_0, T_mult=1, eta_min=cfg.CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = .CustomEfficient(cfg.CFG.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=cfg.CFG.lr,
                     weight_decay=cfg.CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # apex
    # ====================================================
    if cfg.CFG.apex:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1', verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = 0.
    best_loss = np.inf

    for epoch in range(cfg.CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion,
                            optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_col].values

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(valid_labels, preds.argmax(1))

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score}')

        if score > best_score:
            best_score = score
            LOGGER.info(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')

    check_point = torch.load(
        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    valid_folds[[str(c) for c in range(5)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds
