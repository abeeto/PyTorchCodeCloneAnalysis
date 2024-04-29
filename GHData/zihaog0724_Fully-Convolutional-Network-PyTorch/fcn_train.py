import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from fcn_dataset import SegDataset, img_transforms
from fcn_model import fcn_8x_resnet34
from utils import label_accuracy_score

device = torch.device("cuda")

train_dataset = SegDataset(True, img_transforms)
val_dataset = SegDataset(False, img_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

model = fcn_8x_resnet34()
model.to(device)

num_epochs = 50
learning_rate = 1e-3
momentum = 0.7
criterion = nn.BCELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    total_train_mean_iu = 0
    for i, (train_images, train_labels) in enumerate(train_loader):
        train_images = train_images.float().to(device)
        train_labels = train_labels.float().to(device)
        
        optimizer.zero_grad()
        
        output = model(train_images)
        output = torch.sigmoid(output)
        
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        output = output.max(dim=1)[1].data.cpu().numpy()
        train_labels = np.argmax(train_labels.data.cpu().numpy(), axis=1)
        acc, mean_iu = label_accuracy_score(train_labels, output)
        total_train_acc += acc
        total_train_mean_iu += mean_iu

    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        total_val_acc = 0
        total_val_mean_iu = 0
        for val_image, val_label in val_loader:
            val_image = val_image.float().to(device)
            val_label = val_label.float().to(device)
            
            val_output = model(val_image)
            val_output = torch.sigmoid(val_output)
            
            val_loss = criterion(val_output, val_label)
            total_val_loss += val_loss.item()
            
            val_output = val_output.max(dim=1)[1].data.cpu().numpy()
            val_label = np.argmax(val_label.data.cpu().numpy(), axis=1)
            acc, mean_iu = label_accuracy_score(val_label, val_output)
            total_val_acc += acc
            total_val_mean_iu += mean_iu

        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = total_train_acc / len(train_loader)
        epoch_train_mean_iu = total_train_mean_iu / (len(train_loader))

        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_acc = total_val_acc / len(val_loader)
        epoch_val_mean_iu = total_val_mean_iu / len(val_loader)

        print("Epoch {}:, Train loss: {:.4f}, Train acc: {:.4f}, Train MeanIU: {:.4f}; Val loss: {:.4f}, Val acc: {:.4f}, Val MeanIU: {:.4f}".format(epoch+1, epoch_train_loss, epoch_train_acc, epoch_train_mean_iu, epoch_val_loss, epoch_val_acc, epoch_val_mean_iu))
        
        save_path = "./logs/Epoch{}TrainAcc{:.4f}ValAcc{:.4f}.pth".format(epoch+1, epoch_train_acc, epoch_val_acc)
        torch.save(model.state_dict(), save_path)
    scheduler.step()
