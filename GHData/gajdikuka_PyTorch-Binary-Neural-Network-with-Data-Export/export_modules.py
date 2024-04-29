import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

#Saves the model paramaeters
def SaveModel(model, outfile):
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.nan)
    
    mean0Arr = model.bn0.running_mean.cpu().detach().numpy()
    mean1Arr = model.bn1.running_mean.cpu().detach().numpy()
    mean2Arr = model.bn2.running_mean.cpu().detach().numpy()
    mean3Arr = model.bn3.running_mean.cpu().detach().numpy()
    mean4Arr = model.bn4.running_mean.cpu().detach().numpy()
    meanList = [mean0Arr, mean1Arr, mean2Arr, mean3Arr, mean4Arr]
    
    std0Arr = np.sqrt(model.bn0.running_var.cpu().detach().numpy())
    std1Arr = np.sqrt(model.bn1.running_var.cpu().detach().numpy())
    std2Arr = np.sqrt(model.bn2.running_var.cpu().detach().numpy())
    std3Arr = np.sqrt(model.bn3.running_var.cpu().detach().numpy())
    std4Arr = np.sqrt(model.bn4.running_var.cpu().detach().numpy())
    stdList = [std0Arr, std1Arr, std2Arr, std3Arr, std4Arr]
    
    weightList = []
    bnScaleList = []
    bnShiftList = []
    biasList = []
    paramList = [bnScaleList, bnShiftList, weightList, biasList]
    
    i=4;
    for param in model.parameters():
        paramlistIndex = i%4
        paramList[paramlistIndex].append(param.data.cpu().numpy())
        i+=1;
    
    np.savez(outfile, alphas=bnScaleList, gammas=bnShiftList, mus=meanList, sigmas=stdList, As=weightList, bs=biasList)

#Saves 20 correct images for each label
def SaveInput(model, outfile):
    data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])), batch_size=1, shuffle=False)
    counter = [0,0,0,0,0,0,0,0,0,0]
    images = []
    labels = []
    model.eval()
    for data, target in data_loader:
        with torch.no_grad():
            data, target = Variable(np.multiply(data, 255).cuda()), Variable(target)
        pred = model(data).data.max(1, keepdim=True)[1]
        corr = target.flatten().tolist()[0]
        if pred == corr and counter[pred] < 20:
            images.append(data.flatten().squeeze().tolist())
            labels.append(corr)
            counter[corr] +=1
    np.savez(outfile, images=images, labels=labels)