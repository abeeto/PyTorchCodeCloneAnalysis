import torch
import torch.nn as nn
from torchvision import models
import torchvision
import numpy as np
from vision_dataset_method import *
from torch.utils.data import Dataset, DataLoader
import tqdm
from codecarbon import EmissionsTracker
from torchsummary import summary
from pthflops import count_ops

tracker = EmissionsTracker()



ds = ImageClassificationDataset(100, (3, 100, 100), 3)
dl = DataLoader(ds, batch_size = 32)
input_size = (3, 100, 100)


model = models.resnet18(pretrained=True)

num_classes = 3

# Freezing the weights
for param in model.parameters():
    param.required_grad = False


# Replacing the final layer
model.fc =  nn.Sequential(nn.Linear(512, 256), 
                         nn.ReLU(), 
                         nn.Dropout(p=0.5), 
                         nn.Linear(256, num_classes), 
                         nn.LogSoftmax(dim=1))


summary(model, input_size)
count_ops(model, torch.rand(1, *input_size))


optimizer    =  torch.optim.Adam(model.fc.parameters(), lr=0.001)
criterion    =  nn.NLLLoss()

tracker = EmissionsTracker()

tracker.start()
for epoch in range(1):

    model.train() # This sets the model back to training after the validation step
    print('\nEpoch Number {}'.format(epoch+1))

    training_loss = 0.0
    correct_preds = 0
    best_acc = 0
    total = 0

    train_data_loader = tqdm.tqdm(dl)

    for i, j in train_data_loader:
        outputs = model(i)
        optimizer.zero_grad()

        loss = criterion(outputs, j)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    #epoch_loss = training_loss / dataset_size['train']
tracker.stop()
