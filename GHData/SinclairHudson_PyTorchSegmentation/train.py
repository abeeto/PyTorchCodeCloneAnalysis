import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import numpy as np
import os
import segmentation_models_pytorch as smp
from A2D2Segmentation import AudiSegmentationDataset
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from stupidNet import *
from watoNet import *
from WeightedFocalLoss import FocalLoss
import torchvision.transforms as transforms

matplotlib.use(
    'Agg')  # needed because otherwise it tries to actually show the plot. Or something. Matplotlib doesnt make sense.
sns.set()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multi_gpu = True

learning_map = {
    "Car 1": 0,
    "Car 2": 0,
    "Car 3": 0,
    "Car 4": 0,
    "Bicycle 1": 0,
    "Bicycle 2": 0,
    "Bicycle 3": 0,
    "Bicycle 4": 0,
    "Pedestrian 1": 0,
    "Pedestrian 2": 0,
    "Pedestrian 3": 0,
    "Truck 1": 0,
    "Truck 2": 0,
    "Truck 3": 0,
    "Small vehicles 1": 0,
    "Small vehicles 2": 0,
    "Small vehicles 3": 0,
    "Traffic signal 1": 0,
    "Traffic signal 2": 0,
    "Traffic signal 3": 0,
    "Traffic sign 1": 0,
    "Traffic sign 2": 0,
    "Traffic sign 3": 0,
    "Utility vehicle 1": 0,
    "Utility vehicle 2": 0,
    "Sidebars": 0,
    "Speed bumper": 0,
    "Curbstone": 0,
    "Solid line": 1,
    "Irrelevant signs": 0,
    "Road blocks": 0,
    "Tractor": 0,
    "Non-drivable street": 0,
    "Zebra crossing": 1,
    "Obstacles / trash": 0,
    "Poles": 0,
    "RD restricted area": 0,
    "Animals": 0,
    "Grid structure": 0,
    "Signal corpus": 0,
    "Drivable cobblestone": 0,
    "Electronic traffic": 0,
    "Slow drive area": 0,
    "Nature object": 0,
    "Parking area": 0,
    "Sidewalk": 0,
    "Ego car": 0,
    "Painted driv. instr.": 0,
    "Traffic guide obj.": 0,
    "Dashed line": 1,
    "RD normal street": 0,
    "Sky": 0,
    "Buildings": 0,
    "Blurred area": 0,
    "Rain dirt": 0,
}
our_classes = ["background", "Line"]
our_colours = np.array([[0, 0, 0], [0, 0, 255]])
assert len(our_classes) == len(our_colours)
conf = {
    "epochs": 150,
    "learning_rate": 0.0004,
    "momentum": 0.9,
    "batch_size": 2,
    "weight_decay": 0.0001,
    # "weight_balance": [1.03460231848, 276.939057959, 107.45530397, 124.671731118, 87.0446761785, 981.810059761],
    # "weight_balance": [1, 1, 1, 1, 1, 1],
    "weight_balance": [1, 50],
    # "weight_balance": [1, 50],
    "backbone": "efficientnet-b1",
    "positional_encoding": True,
    "loss_function": "Focal"
}
wandb.init(project="wato-roadline-segmentation", config=conf)


def calculateIoU(conf_matrix):
    per_class_iou = []
    for x in range(conf_matrix.shape[0]):
        union = np.sum(conf_matrix[:, x]) + np.sum(conf_matrix[x, :]) - conf_matrix[x, x]
        intersection = conf_matrix[x, x]
        if union != 0:
            iou = intersection / union
        else:
            iou = 0
        per_class_iou.append(iou)
    return per_class_iou


best_IoU = 0.0
best_accuracy = 0.0
step = 0
n_classes = len(our_classes)
assert n_classes == len(conf["weight_balance"])

trainset = AudiSegmentationDataset("/mnt/sda/datasets/Audi/camera_lidar_semantic", learning_map,
                                   positional=conf["positional_encoding"])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf["batch_size"], shuffle=True)

testset = AudiSegmentationDataset("/mnt/sda/datasets/Audi/camera_lidar_semantic", learning_map, split="val",
                                  positional=conf["positional_encoding"])
testloader = torch.utils.data.DataLoader(testset, batch_size=conf["batch_size"], shuffle=False)

in_channels = 3
if conf["positional_encoding"]:
    in_channels += 2
if conf["backbone"] == "WatoNet":
    model = WatoNet(in_channels, n_classes).to(device)
else:
    model = smp.Unet(conf["backbone"], encoder_weights='imagenet', in_channels=in_channels, classes=n_classes,
                     activation='softmax').to(device)
if multi_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
# model = WatoNet(n_classes=6, in_channels=in_channels).to(device)
# model.load_state_dict(torch.load("path"))
wandb.watch(model)
if conf["loss_function"] == "Xentropy":
    criterion = nn.NLLLoss(weight=torch.Tensor(conf["weight_balance"]).to(device))
elif conf["loss_function"] == "Focal":
    criterion = FocalLoss(weight=torch.Tensor(conf["weight_balance"]).to(device), gamma=2)

optimizer = Adam(model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
for epoch in range(conf["epochs"]):
    if epoch % 2 == 0:
        print("Running Validation")
        conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        for batch, (images, labels) in enumerate(testloader):
            # cut early to get more training in!
            if batch > 80:
                break
            model.eval()
            yhat = model(images.to(device))
            confidence, predictions = torch.max(yhat, 1)
            # print(predictions.size())
            confidence = confidence.cpu().detach().numpy()
            if batch < 3:
                blankchannel = np.zeros(confidence.shape)
                confidence_image = np.stack((confidence * 255, blankchannel, blankchannel), axis=-1)
                # print(confidence_image.shape)
                prediction_images = [our_colours[p] for p in predictions.cpu().int()]
                label_images = [our_colours[p] for p in labels.cpu().int()]
                camera_images = (images[:, :3, :, :] * 128) + 128  # from [-1, 1] to [0, 255]

                prediction_images = np.array(prediction_images)
                label_images = np.array(label_images)
                camera_images = np.moveaxis(camera_images.int().numpy(), 1, -1)
                bigprediction = np.concatenate((camera_images, label_images, prediction_images, confidence_image),
                                               axis=1)
                wandb.log({"prediction_" + str(batch): [wandb.Image(img) for img in bigprediction]}, step=step)

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            cm = confusion_matrix(predictions.reshape((-1)), labels.reshape((-1)), labels=list(range(n_classes)))
            conf_matrix += cm
            print(f"VALIDATION epoch: {epoch}, batch: {batch}")

        test_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
        if test_accuracy > best_accuracy and test_accuracy > 0.5:
            torch.save(model.state_dict(),
                       os.path.join(wandb.run.dir, f"{conf['backbone']}-{epoch}-{test_accuracy}.pth"))
            best_accuracy = test_accuracy
        plt.clf()
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(np.log(conf_matrix + 1), annot=False, fmt="d", linewidths=0.0, ax=ax,
                    xticklabels=our_classes, yticklabels=our_classes)
        plt.tight_layout()
        ious = calculateIoU(conf_matrix)
        wandb.log({"Confusion_matrix": wandb.Image(plt),
                   "test_accuracy": test_accuracy,
                   "IoUs": ious,
                   "mIoU": sum(ious) / n_classes,
                   }, step=step)

    for batch, (images, labels) in enumerate(trainloader):
        model.train()
        yhat = model(images.to(device))  # returns probabilities of each class, it's softmaxed
        loss = criterion(torch.log(torch.clamp(yhat.float(), 1e-6, 1.0)), labels.long().to(device))
        print(f"TRAIN epoch: {epoch}, batch: {batch}, loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item()}, step=step)

        step += 1
