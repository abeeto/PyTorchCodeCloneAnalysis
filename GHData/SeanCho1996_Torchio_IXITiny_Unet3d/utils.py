from io import BytesIO
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import numpy as np
import os
import torchio as tio
from tqdm import tqdm
import uuid

from dataset import Datasets3D, read_param
from model import UNet


SPATIAL_DIMENSIONS = 2, 3, 4


def prediction(state_dict, num_classes, img_file, device, landmarks, batch_size):
    
    subject_embedding = tio.Subject(
                mri=tio.ScalarImage(img_file)
                )

    validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad((48, 60, 48)),
            tio.HistogramStandardization({'mri': landmarks}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.OneHot(),
        ])

    test_data = validation_transform(subject_embedding)

    patch_size = 48, 48, 48  # we can user larger patches for inference
    patch_overlap = 4, 4, 4
    grid_sampler = tio.inference.GridSampler(
        test_data,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=2*batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler)

    model = UNet(num_classes)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = DataParallel(model)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['mri'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            probabilities = model(inputs).softmax(dim=1)
            aggregator.add_batch(probabilities, locations)

    output_tensor = aggregator.get_output_tensor()
    output_tensor = torch.argmax(output_tensor, dim=0).unsqueeze(0)

    affine = test_data.mri.affine
    prediction = tio.LabelMap(tensor=output_tensor.short(), affine=affine)

    spatial_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4)
    ])
    resample_shape = spatial_transform(tio.ScalarImage(img_file)).shape[-3:]

    print(f"raw: {prediction}")
    prediction = tio.CropOrPad(resample_shape)(prediction)
    print(f"crop/pad: {prediction}")
    prediction = tio.Resample(img_file, image_interpolation='nearest')(prediction)
    print(prediction)

    save_name = f'res_{uuid.uuid4().hex}.nii.gz'
    prediction.save(save_name)

    with open(save_name, "rb") as f:
        buffered = BytesIO(f.read())

    return buffered

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score


def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)


def train_one_epoch(train_loader, model, device, optimizer):
    epoch_losses = []

    model.train()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        inputs = batch['mri'][tio.DATA].to(device)
        targets = batch['brain'][tio.DATA].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()

            batch_loss.backward()
            optimizer.step()

            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    
    return epoch_losses.mean()


def valid(test_loader, model, device):
    epoch_losses = []
    val_acc = []

    model.eval()
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        inputs = batch['mri'][tio.DATA].to(device)
        targets = batch['brain'][tio.DATA].to(device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)
            batch_losses = get_dice_loss(probabilities, targets)
            dice_score = get_dice_score(probabilities, targets)

            batch_loss = batch_losses.mean()
            acc = dice_score.mean()

            epoch_losses.append(batch_loss.item())
            val_acc.append(acc.item())
    epoch_losses = np.array(epoch_losses)
    val_acc = np.array(val_acc)
    
    return epoch_losses.mean(), val_acc.mean()