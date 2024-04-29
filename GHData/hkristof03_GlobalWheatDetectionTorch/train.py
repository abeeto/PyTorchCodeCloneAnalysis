import torch
import pandas as pd
import numpy as np
# Timing utility
from timeit import default_timer as timer

from utils.config_parser import parse_args, parse_yaml
import dataloader as dl
from transformations import transforms as trfs
from models.model_zoo import get_model
from utils.averager import Averager

from utils.metrics import calculate_image_precision


def train_model(
    train_data_loader,
    valid_data_loader,
    model,
    optimizer,
    num_epochs,
    lr_scheduler=None,
    path_save_model='./artifacts/saved_models/fasterrcnn_test.pth'
    ):
    """
    """
    scores_dict_train = get_empty_scores_dict(train_data_loader)
    scores_dict_valid = get_empty_scores_dict(valid_data_loader)
    history = []

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cpu_device = torch.device('cpu')

    model.to(device)

    overall_start = timer()

    n_train_batches = len(train_data_loader)
    # Main loop
    for epoch in range(num_epochs):

        # Keep track of training loss and validation MAP each epoch
        train_loss = 0.0
        # Set to training
        model.train()
        start = timer()

        for ii, (images, targets, image_ids) in enumerate(train_data_loader):

            print(f"\nEpoch #{epoch} Train Batch #{ii}/{n_train_batches}")

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            #print(f"Loss dict: {loss_dict}")
            losses = sum(loss for loss in loss_dict.values())
            #print(f"Losses: {losses}")
            loss_value = losses.item()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss_value * len(images)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"\nEpoch #{epoch}: {timer() - start:.2f} seconds elapsed.")

        scores_dict_train, train_map = predict_data_set(model,
            train_data_loader, scores_dict_train, epoch, device, cpu_device,
            'Train')

        scores_dict_valid, valid_map = predict_data_set(model,
            valid_data_loader, scores_dict_valid, epoch, device, cpu_device,
            'Validation')

        # Calculate average losses
        train_loss = train_loss / len(train_data_loader.dataset)

        history.append([train_loss, train_map, valid_map])

    # End of training
    total_time = timer() - overall_start
    print(
        f"{total_time:.2f} total seconds elapsed. "
        f"{total_time / (epoch + 1):.2f} seconds per epoch"
    )
    torch.save(model.state_dict(), path_save_model)

    df_history = pd.DataFrame(
        history,
        columns=['train_loss', 'train_map', 'valid_map']
    )
    df_scores_train = pd.DataFrame(scores_dict_train)
    df_scores_valid = pd.DataFrame(scores_dict_valid)

    return (df_history, df_scores_train, df_scores_valid)


def predict_data_set(
    model,
    data_loader,
    scores_dict,
    epoch,
    device,
    cpu_device,
    dataset,
    ):
    """
    """
    iou_thresholds = [np.round(x, 2) for x in np.arange(0.5, 0.76, 0.05)]
    image_precisions = []
    n_batches = len(data_loader)

    # Don't need to keep track of gradients
    with torch.no_grad():

        if model.training:
            # Set to evaluation mode (BatchNorm and Dropout works differently)
            model.eval()

        # Validation loop
        for ii, (images, targets, image_ids) in enumerate(data_loader):

            print(
                f"\nEpoch #{epoch} {dataset} Batch #{ii}/{n_batches} "
                f"calculating Mean Average Precision..."
            )
            # Tensors to gpu
            images = list(image.to(device) for image in images)

            outputs = model(images)
            outputs = [
                {k: v.to(cpu_device).numpy() for k, v in t.items()} for t in outputs
            ]

            for idx, image in enumerate(images):

                preds = outputs[idx]['boxes']
                scores = outputs[idx]['scores']
                gt_boxes = targets[idx]['boxes'].numpy()

                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = preds[preds_sorted_idx]

                image_precision = calculate_image_precision(
                    gt_boxes,
                    preds_sorted,
                    iou_thresholds,
                    'coco',
                )
                image_precisions.append(image_precision)

                image_id = image_ids[idx]
                scores_dict[image_id].append(image_precision)

    map = np.mean(image_precisions)
    print(f"{dataset} MAP: {round(map, 4)}")

    return scores_dict, map


def get_empty_scores_dict(data_loader):
    """
    """
    scores_dict = {}

    for ii, (images, targets, image_ids) in enumerate(data_loader):

        for idx, image_id in enumerate(image_ids):

            scores_dict[image_id] = []

    return scores_dict


def train_random_holdout(config: dict):
    """"""
    psm = configs_train['path_save_model']
    mn = configs_train['model_name']
    ph = configs_train['path_history']
    path_history = ph + mn
    config_dataloader = config['dataloader']
    path_df = config_dataloader['path_df']

    df = dl.transform_df(path_df)
    train_df, valid_df = dl.get_train_valid_df(path_df, valid_size=0.1)
    train_trf = trfs.ImgAugTrainTransform()
    valid_trf = trfs.to_tensor()

    train_data_loader, valid_data_loader = dl.get_train_valid_dataloaders(
        config_dataloader, train_df, valid_df, dl.collate_fn,
        train_trf, valid_trf)

    model = get_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9,
        weight_decay=0.0005)
    lr_scheduler = None
    num_epochs = configs_train['epochs']

    df_history, df_scores_train, df_scores_valid = train_model(train_data_loader,
        valid_data_loader, model, optimizer, num_epochs, lr_scheduler,
        path_save_model)

    df_history.to_csv(path_history + '_history.csv', index=False)
    df_scores_train.to_csv(path_history + '_scores_train.csv', index=False)
    df_scores_valid.to_csv(path_history + '_scores_valid.csv', index=False)


def train_skfold(config: dict):
    """"""
    configs_train = config['train']
    psm = configs_train['path_save_model']
    mn = configs_train['model_name']
    ph = configs_train['path_history']
    path_history = ph + mn
    config_dataloader = config['dataloader']
    path_df = config_dataloader['path_df']

    df = dl.transform_df(path_df)
    df = dl.split_stratifiedKFolds_bbox_count(df, config_dataloader['n_splits'])
    folds = list(df['fold'].unique())
    train_trf = trfs.ImgAugTrainTransform()
    valid_trf = trfs.to_tensor()

    for i, fold in enumerate(folds):

        print(f"{'_'*30}Training on fold {fold}...{'_'*30}")
        path_save_model = psm + mn + f'_fold{fold}.pth'
        train_df, valid_df = dl.get_train_valid_df_skfold(df, fold)
        train_data_loader, valid_data_loader = dl.get_train_valid_dataloaders(
            config_dataloader, train_df, valid_df, dl.collate_fn,
        train_trf, valid_trf)

        model = get_model()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9,
            weight_decay=0.0005)
        lr_scheduler = None
        num_epochs = configs_train['epochs']

        df_history, df_scores_train, df_scores_valid = train_model(
            train_data_loader, valid_data_loader, model, optimizer, num_epochs,
            lr_scheduler, path_save_model
        )
        df_history.to_csv(path_history + f'fold{fold}_history.csv', index=False)
        df_scores_train.to_csv(path_history + f'fold{fold}_scores_train.csv',
            index=False)
        df_scores_valid.to_csv(path_history + f'fold{fold}_scores_valid.csv',
            index=False)



if __name__ == '__main__':

    args = parse_args()
    config = parse_yaml(args.pyaml)

    if config['dataloader']['stratifiedKFold']:
        train_skfold(config)
    else:
        train_random_holdout(config)
