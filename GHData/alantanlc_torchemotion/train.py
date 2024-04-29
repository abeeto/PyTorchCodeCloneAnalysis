import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets.IemocapDataset import IemocapDataset
from models.VGG_convnet import VGG_convnet
from models.DNN import DNN

import time
import copy
from scipy import stats

def compute_number_of_corrects(preds, data, n_frames):
    # Compute number of corrects for variable batch size
    n_corrects = 0
    for i in range(len(n_frames)):
        start_idx = 0 if i == 0 else torch.sum(n_frames[:i]).long()
        end_idx = n_frames[i].long()
        target_label = data[start_idx]
        predicted_label = stats.mode(preds[start_idx:start_idx+end_idx]).mode[0]
        n_corrects += (target_label == predicted_label)
    return n_corrects

def train_model_vgg(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            total_n_frames = 0

            # Iterate over data.
            for inputs, emotions, n_frames in dataloaders[phase]:
                inputs = inputs.to(device)
                emotions = emotions.long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, emotions)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == emotions.data)
                running_corrects += compute_number_of_corrects(preds, emotions.data, n_frames)

                total_n_frames += inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / total_n_frames
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load train and valid datasets
datasets = {
    'train': IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release', sessions=[1, 2, 4, 5]),
    'val': IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release', sessions=[3])
}
dataset_sizes = { x: len(datasets[x]) for x in ['train', 'val'] }
dataloaders = { x: torch.utils.data.DataLoader(datasets[x], batch_size=2, shuffle=True, num_workers=4, collate_fn=IemocapDataset.collate_fn) for x in ['train', 'val'] }
# dataloaders = { x: torch.utils.data.DataLoader(datasets[x], batch_size=128, shuffle=(x=='train'), num_workers=4, collate_fn=IemocapDataset.collage_fn_vgg) for x in ['train', 'val'] }

# Model
model_ft = DNN(400, 1000, 1500, 9)
# model_ft = VGG_convnet()
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train and evaluate
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
# model_ft = train_model_vgg(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)