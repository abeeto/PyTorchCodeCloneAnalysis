import torch, os, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from dataloader import get_loader
from dataset import get_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, roc_auc_score, recall_score, precision_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset = get_dataset(IMG_SIZE=224)
train_dataloader, val_dataloader, test_dataloader = get_loader(BATCH_SIZE=128)

# model_ft = torch.load('./models/model_resnet50.pt')
model_ft = torch.load('./models/model_mobilenetv2.pt')

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([{'params':model_ft.fc.parameters()}], lr=0.001)
optimizer = torch.optim.Adam([{'params':model_ft.classifier[1].parameters()}], lr=0.001)


nb_classes = len(train_dataset.classes)
confusion_matrix = np.zeros((nb_classes, nb_classes))

def val(model, device, loader, dataset):
    model.eval()
    test_loss, correct = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, data in enumerate(loader):       
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

            y_pred.extend(pred.view(-1).detach().cpu().numpy())
            y_true.extend(y.view(-1).detach().cpu().numpy())
            for t, p in zip(y.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    test_loss /= len(loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataset),
        100. * correct / len(dataset)))

    plt.figure(figsize=(15,10))

    class_names = train_dataset.classes
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

    return y_true, y_pred


y_true, y_pred = val(model=model_ft, device=DEVICE, loader=test_dataloader, dataset=test_dataset)

print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True, digits=2)

pd_report = pd.DataFrame(report).transpose().round(2)

# pd_report.to_csv('./results/summary_mobilenetv2.csv')
# pd_report.to_csv('./results/summary_resnet50.csv')