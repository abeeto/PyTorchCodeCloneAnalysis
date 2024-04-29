import os

from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch
import collections

from matplotlib import pyplot as plt
from utils import initialize_model, train_model

train_data_path = "Data/train"
test_data_path = "Data/test"
batch_size = 2
gpu_id = int(input("GPU id:"))
device = torch.device("cuda:%d" % gpu_id)
num_epochs = 10
pretrained = False
fc_only = False
model_name = input("model name:")
iter = int(input("model index:"))
num_classes = 2

# Load model
model_ft, input_size = initialize_model(model_name, num_classes, fc_only, pretrained)
model_ft.to(device)

# Load dataset
train_ds = ImageFolder(train_data_path, transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

val_ds = ImageFolder(test_data_path, transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

print(collections.Counter(train_ds.targets))
print(collections.Counter(val_ds.targets))

dataloaders = {}
dataloaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dataloaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss(reduction="mean")
opt = optim.Adam(model_ft.parameters(), lr=1e-3)
lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)

model_ft, acc_history, loss_history = train_model(model_ft, dataloaders, criterion, opt, num_epochs, model_name=="inception", device)

# Create Save folder
os.makedirs(model_name, exist_ok=True)

# Save model as file
torch.save(model_ft.state_dict(), "%s/%s%d.pt" % (model_name, model_name, iter))

# Plot acc, loss
plt.plot(loss_history['train'])
plt.plot(loss_history['val'])
plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig("%s/%s_loss%d.png" % (model_name, model_name, iter))
plt.clf()

plt.plot(acc_history['train'])
plt.plot(acc_history['val'])
plt.title('model training loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig("%s/%s_accuracy%d.png" % (model_name, model_name, iter))
plt.clf()

with open("%s/%s_%d.log" % (model_name, model_name, iter)) as f:
    train_acc = acc_history["train"]
    val_acc = acc_history["val"]
    train_loss = loss_history["train"]
    val_loss = loss_history["val"]
    for tacc, vacc, tloss, vloss in zip(train_acc, val_acc, train_loss, val_loss):
        f.write("train acc: %.6f\t train loss: %.6f\t val acc: %.6f\t val loss: %.6f\n" % (tacc, tloss, vacc, vloss))
    