from torchvision import models
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from matplotlib import pyplot as plt
import copy, os, argparse, pathlib
import numpy as np
import itertools
import sklearn.metrics as metrics

from covid import get_data_loader

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

    current_lr = get_lr(opt)
    print('current lr=%d' % (current_lr))

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

def loss_epoch(model, device, loss_func, data_loader, sanity=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(data_loader.dataset)

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b
        if sanity is True:
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric

def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["opt"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity = params["sanity"]
    lr_scheduler = params["lr_scheduler"]
    path_to_weights = params["path_to_weights"]
    device = params["device"]
    
    loss_history = {
            "train": [],
            "val": [],
    }
    metric_history = {
            "train": [],
            "val": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print("Epoch %d/%d, current lr = %f" % (epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, device, loss_func, train_dl, sanity, opt)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, device, loss_func, val_dl, sanity)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path_to_weights)
            print("Copied best model weights!")
 
        lr_scheduler.step()
        print("train loss: %.6f, val loss: %.6f, train accuracy: %.2f, val accuracy: %.2f" %
                (train_loss, val_loss, 100*train_metric, 100*val_metric))

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

def test(model, data_loader, device, output_folder):
    metric = 0.0
    len_data = len(data_loader.dataset)
    confusion_matrix = torch.tensor([[0, 0],[0, 0]], dtype=torch.int64)
    roc_y = []
    roc_score = []

    for xb, yb in data_loader:
        roc_y += yb.tolist()

        g_xb = xb.to(device)
        g_yb = yb.to(device)
        output = model(g_xb)
        pred = output.argmax(dim=1, keepdim=True)
        tmp = pred.eq(g_yb.view_as(pred)).sum().item()
        metric += tmp
        print("batch result: %d/%d %f" % (tmp, len(xb), tmp / len(xb) * 100))

        #Updata confusion matrix
        for actual, predicted in zip(yb, pred):
            confusion_matrix[actual][predicted] += 1

        for idx, prob in enumerate(output):
            roc_score.append(prob[1].cpu().detach().data)

    curve = metrics.roc_curve(roc_y, roc_score)
    roc_auc = metrics.auc(curve[0], curve[1])

    plt.title('Receiver Operating Characteristic')
    plt.plot(curve[0], curve[1], 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig(os.path.join(output_folder, "roc.png"), bbox_inches = "tight")
    plt.clf()

    print("Total accuracy : %f(%d/%d)" % (metric / len_data * 100, metric, len_data))
    print("\t\t Actual 0\t\t Actual 1")
    print("Pred 0\t\t %d\t\t\t %d" % (confusion_matrix[0][0], confusion_matrix[1][0]))
    print("Pred 1\t\t %d\t\t\t %d" % (confusion_matrix[0][1], confusion_matrix[1][1]))
    
    return confusion_matrix

def load_model(model_name, pretrained, fc_only, device):
    if model_name == "resnet":
        model = models.resnet18(pretrained=pretrained)
        if fc_only:
            for param in model.parameters():
                param.requires_grad = False
    
        num_classes = 2
        num_ftrs= model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg":
        model = models.vgg16(pretrained=pretrained)
        if fc_only:
            for param in model.parameters():
                param.requires_grad = False

        num_classes = 2
        num_ftrs= model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    else:
        print("Undefined model name")
        exit(1)
  
    model.to(device)
 
    return model

def draw_result(title, name, log ,num_epochs):
    plt.title("Train-Val " + title)
    plt.plot(range(1,num_epochs+1), log["train"], label="train")
    plt.plot(range(1,num_epochs+1), log["val"], label="val")
    plt.ylabel(title)
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_folder, name))
    plt.clf()

def plot_confusion_matrix(cm, classes, output, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i][j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output, "confusion.png"),bbox_inches = "tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prameter for ML')
    parser.add_argument('-O', type=pathlib.Path, help='Output folder path', required=True)
    parser.add_argument('-I', type=pathlib.Path, help='input image folder path', required=True)
    parser.add_argument('-M', choices=['resnet', 'vgg'], required=True)
    parser.add_argument('-P', choices=['T', 'F'], required=True)
    parser.add_argument('-p', choices=['T', 'F'], required=True)
    parser.add_argument('-E', type=int, required=True)
    parser.add_argument('-B', type=int, required=True)
    parser.add_argument('-m', type=int, required=True)
    parser.add_argument('-g', type=int, required=True)
    
    parsed = parser.parse_args()

    output_folder = str(parsed.O)
    input_data = str(parsed.I)
    model_name = parsed.M
    pretrained = parsed.P == 'T'
    fc_only = parsed.p == 'T'
    num_epochs = parsed.E
    batch_size = parsed.B
    max_data = parsed.m
    gpu_id = parsed.g
    
    try:
        os.makedirs(output_folder, exist_ok=False)
    except FileExistsError:
        print("Overwriting output folder!")
    
    with open(os.path.join(output_folder,"cmdline.log"), "w") as f:
        f.write("input data: %s\n" % input_data)
        f.write("model name: %s\n" % model_name)
        f.write("Pretrained : %s\n" % str(pretrained))
        f.write("Gradient only linear : %s\n" % str(fc_only))
        f.write("num epochs: %d\n" % num_epochs)
        f.write("batch size: %d\n" % batch_size)
        f.write("max data: %d\n" % max_data)

    device = torch.device("cuda:%d" % gpu_id)
    # Load model and data
    model = load_model(model_name, pretrained, fc_only, device)
    train_dl, val_dl, test_dl = get_data_loader(input_data, batch_size, output_folder, max_data)
    
    # Set loss, optimizer, learning late scheduler
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)
    
    params_train = {
            "num_epochs" : num_epochs,
            "opt" : opt,
            "loss_func" : loss_func,
            "train_dl" : train_dl,
            "val_dl" : val_dl,
            "sanity" : False,
            "lr_scheduler" : lr_scheduler,
            "path_to_weights" : os.path.join(output_folder, "weight.pt"),
            "device" : device
            }

    model, loss_hist, metric_hist = train_val(model, params_train)

    draw_result("Loss", "Train-val-loss.png", loss_hist, num_epochs)
    draw_result("Accuracy", "Train-val-Accuracy.png", metric_hist, num_epochs)

    print("############### Test Phase ###############")
    cf = test(model, test_dl, device, output_folder)
    plot_confusion_matrix(cf, ["non Covid19", "Covid19"], output_folder)

    
