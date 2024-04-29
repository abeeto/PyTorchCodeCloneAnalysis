# import all we need from the standard python toolset
import os
import shutil
import warnings
import argparse
import random
warnings.filterwarnings('ignore')

# import usefull tools
import csv
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from PIL import Image, ImageTk

# import torch and tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch_optimizer as optimNew

import torchvision
from torchvision import datasets,transforms 
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics

from MultiLabelDataSet import *

from tensorboardX import SummaryWriter
import configparser
import json
config = configparser.ConfigParser()

parser = argparse.ArgumentParser(description='MyPyTorch Network')
parser.add_argument('--debugLevel',      '-d', type=int,  default=0,                  required=False, help="Debug Output level")
parser.add_argument('--doTrain',         '-t',            default=False,              required=False, help="Run Training",               action='store_true')
parser.add_argument('--doTest',          '-v',            default=False,              required=False, help="Run Validation Test",        action='store_true')
parser.add_argument('--batchSize',       '-b', type=int,  default=4,                  required=False, help='Batch Size')
parser.add_argument('--batchLimit',      '-l', type=int,  default=0,                  required=False, help='Batch Limit')
parser.add_argument('--epochs',          '-e', type=int,  default=10,                 required=False, help='Num of Epochs')
parser.add_argument('--init',            '-i',            default=False,              required=False, help='Init with clean Network',    action='store_true')
parser.add_argument('--FeatureExtract',  '-f',            default=False,              required=False, help='Use NO Feature Extraction',  action='store_true')
parser.add_argument('--noPreTrain',      '-np',           default=True,               required=False, help='Use NO pre-trained Network', action='store_false')
parser.add_argument('--showData',        '-s',            default=False,              required=False, help='Show a set of images',       action='store_true')
parser.add_argument('--configFile',      '-c',            default="myPyTorch.ini",    required=False, help='Configuration File name')
parser.add_argument('--net',             '-n',            default="ResNeXt50",        required=False, help='Neural Network (use list to get list)')
parser.add_argument('--testSplit',       '-ts', type=int, default=10,                 required=False, help='Test Split %')
args = parser.parse_args()
debug           = args.debugLevel
doTrain         = args.doTrain
doTest          = args.doTest
trBatchSize     = args.batchSize
trBatchLimit    = args.batchLimit # 0 means to the full scale of the training set
trMaxEpoch      = args.epochs
doClean         = args.init
usePreTrain     = args.noPreTrain
feature_extract = args.FeatureExtract
doShow          = args.showData
trConfFile      = args.configFile
trNetName       = args.net
trTestSplit     = args.testSplit

try:
    config.read(trConfFile)
    print("Reading configuration from: ", trConfFile)
    conf_suffix = config['CheckPoint']['suffix']
    conf_home = config['Data']['home']
    conf_ipath = config['Data']['ipath']
    conf_ClassNames = json.loads(config['Data']['ClassNames'])
except:
    print("Using default configuration")
    conf_suffix = '_Net.pth'
    conf_home = '../../Test/CRX8/'
    conf_ipath = 'image/'
    conf_ClassNames = ['No Finding','Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
trCheckPoint    = args.net+conf_suffix
trFilePath      = conf_home
trImgPath       = conf_ipath
nnClassNames    = conf_ClassNames

##############################################
# Note: FeatureExtract = True generates ERROR
##############################################
if trNetName == "list":
    print("- MyPyTorch network list ----------------------")
    print("\tResNext50 - ResNext50_32x4")
    print("- MyPyTorch network list ----------------------")
    exit()
else:
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("= MyPyTorch Network ===========================")
    print("Debug level:\t\t",       debug)
    print("Run Training:\t\t",      doTrain)
    print("Run Validation:\t\t",    doTest)
    print("Batch Size:\t\t",        trBatchSize)
    print("Batch Limit:\t\t",       trBatchLimit)
    print("Number of Epochs:\t",    trMaxEpoch)
    print("Init Network:\t\t",      doClean)
    print("PreTrain Net:\t\t",      usePreTrain)
    print("Show Data:\t\t",         doShow)
    print("Neural Network: \t",     trNetName)
    print("Checkpoint File:\t",     trCheckPoint)
    print("Path to Data Files:\t",  trFilePath)
    print("Path to Image Files:\t", trImgPath)
    print("Test Split %:\t\t",      trTestSplit,"%")
    print("\nUsing device:\t\t",    device)
    print("= MyPyTorch Network ===========================")

# define system wide parameters
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

#pathFileTrain = '../../Test/CRX8/Data_Train.csv'
#pathFileValid = '../../Test/CRX8/Data_Valid.csv'
pathFileTrain = trFilePath + 'Data_Train.csv'
pathFileValid = trFilePath + 'Data_Valid.csv'
pathFileImages = trFilePath + trImgPath
#nnClassNames = ['No Finding','Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
#nnClassNames = ['G-0','G-1','G-2','G-3','G-4']

imgtransResize = (320, 320)         # (320,320)
imgtransCrop = 224                  # 224
#imgtransResize = (512, 512)         # (320,320)
#imgtransCrop = 512                  # 224
nnClassCount = len(nnClassNames)     #dimension of the output
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)
print('-- Defining DataSets')

dataset = MultiLabelDataSet(pathFileTrain, pathFileImages, len(nnClassNames), transformSequence, policy="ones")
trSplit = int(len(dataset)*trTestSplit/100)
#datasetTest, datasetTrain = random_split(dataset, [5000, len(dataset) - 5000])
datasetTest, datasetTrain = random_split(dataset, [trSplit, len(dataset) - trSplit])
datasetValid = MultiLabelDataSet(pathFileValid, pathFileImages, len(nnClassNames), transformSequence)
print("-- \tSize of DataSet  : ",len(dataset)," images")
print("-- \tTraining Set Size: ",len(dataset)-trSplit," images")
print("-- \tTest Set Size    : ",trSplit," images")

print('-- Defining DataLoader')
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False)
dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=0)

trMaxBatch = len(datasetTrain)//trBatchSize
trMaxEval = len(datasetValid)//trBatchSize
#debug limit batch runs
if trBatchLimit>0: 
    trMaxBatch = trBatchLimit
    trMaxEval = trBatchLimit
    
print("-- \tbatch size        :", trBatchSize)
print("-- \tMax# of epochs    :", trMaxEpoch)
print("-- \tMax# of batches   :", trMaxBatch)
print("-- \tMax# of evals     :", trMaxEval)


if doShow:
    img_path, images, classes = next(iter(dataLoaderTrain))
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225] 
    def gridshow(inp, title):
        inp = inp.cpu().numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.figure (figsize = (12, 6))
        plt.imshow(inp)
        plt.title(trNetName+' - Batch Sample')
    #    plt.pause(5)  
        plt.show()  
        
    out = torchvision.utils.make_grid(images)
    gridshow(out, title="image sample")
    if debug > 2: print(classes)

# define network model and parameters
print('-- initializing Network')
print("-- \tNetwork:", trNetName)
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
if trNetName == "ResNeXt50":
    model = models.resnext50_32x4d (pretrained=usePreTrain)
    #Redefining the last layer to classify inputs into the nnClassCount classes 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nnClassCount)
    set_parameter_requires_grad(model, feature_extract)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "ResNet18":
    model = models.resnet18(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "ResNet34":
    model = models.resnet34(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "ResNet50":
    model = models.resnet50(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "ResNet101":
    model = models.resnet101(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "ResNet152":
    model = models.resnet152(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "AlexNet":
    model = models.alexnet(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,nnClassCount)    
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "SqueezeNet":
    model = models.squeezenet1_1(pretrained=usePreTrain)
    model.classifier[1] = nn.Conv2d(512, nnClassCount, kernel_size=(1,1), stride=(1,1))
    set_parameter_requires_grad(model, feature_extract)
    model.num_classes = nnClassCount
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "Vgg11":
    model = models.vgg11_bn(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "Vgg13":
    model = models.vgg11_bn(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "Vgg16":
    model = models.vgg16_bn(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "Vgg19":
    model = models.vgg19_bn(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "DenseNet121":
    model = models.densenet121(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "DenseNet161":
    model = models.densenet161(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "DenseNet169":
    model = models.densenet169(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
elif trNetName == "DenseNet201":
    model = models.densenet201(pretrained=usePreTrain)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, nnClassCount)
    criterion = nn.BCELoss(size_average = True)
else:
    print('the network name you have entered is not supported yet, available networks:')
    print('  - ResNeXt50')
    print('  - ResNet18')
    print('  - ResNet34')
    print('  - ResNet50')
    print('  - ResNet101')
    print('  - ResNet152')
    print('  - AlexNet')
    print('  - SqueezeNet')
    print('  - Vgg11')
    print('  - Vgg13')
    print('  - Vgg16')
    print('  - Vgg19')
    print('  - DenseNet121')
    print('  - DenseNet161')
    print('  - DenseNet169')
    print('  - DenseNet201')
    sys.exit()
if debug > 2: print(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
writer = SummaryWriter(log_dir=os.path.join( 'runs', trNetName, timestampLaunch))


def train_model(model, checkpoint, criterion, optimizer, num_epochs=25, maxBatch=50, maxEval=10):

    lossTrain = []
    lossEval = []
    dataStep = 1
    if maxBatch>100:
        dataStep = round(maxBatch/100)
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    if debug > 1: 
        f_csv = open('LossOutput'+timestampLaunch+'.csv','a', newline='')
        fw_csv = csv.writer(f_csv)

    if doClean:
        print ("-- \tusing clean net")
    else:
        if checkpoint != None and use_gpu:
            if os.path.isfile(checkpoint):
                model.load_state_dict(torch.load(checkpoint))
                print ("-- \tLoading checkpoint ", checkpoint)
            else:
                print ("-- \tNo checkpoint file found - using fresh net")
        else:
            print ("-- \tNo checkpoint given - using fresh net")

    model = model.to(device)

    lossMIN = 100000
    
    for epoch in range(num_epochs):
        print('Epoch = ', epoch, " Start Training")

        # train model for this epoch
        model.train()
        
        maxStep = maxBatch
        bar = progressbar.ProgressBar(max_value=maxStep)
        nBatch=0
        lossVal = 0
        chkTotal = 0
        accVal = [0 for i in range(nnClassCount)]
        failVal = [0 for i in range(nnClassCount)]
        confVal = [0 for i in range(nnClassCount)]
        weakVal = [0 for i in range(nnClassCount)]
        accTotal = 0
        failTotal = 0
        confTotal = 0
        weakTotal = 0
        TPVal = [0 for i in range(nnClassCount)]
        TNVal = [0 for i in range(nnClassCount)]
        FPVal = [0 for i in range(nnClassCount)]
        FNVal = [0 for i in range(nnClassCount)]
        outTruth = []
        outPred  = []
        for img_path, images, labels  in (dataLoaderTrain):
            bar.update(nBatch)
            #print(".",end='')
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # moved before call to model
            with torch.set_grad_enabled(True): # as of Finetuning article
                outputs = model(images)
                outputs = outputs.to(device)
                outSigmoid = torch.sigmoid(outputs)
                loss = criterion(outSigmoid,labels)            
                lv = loss.item()
                loss.backward()
                optimizer.step()
                
            writer.add_scalar('Train/loss', lv, epoch*maxStep + nBatch)
            lossVal += lv
            if debug > 1: print(labels)
            if debug > 1: print(outputs)
            batchCnt, Ccnt = labels.size()
            for j in range(batchCnt):
                sglTruth = np.array(labels[j].cpu(), dtype='int')
                sglPred  = np.array(outSigmoid[j].cpu().detach(), dtype='float')
    #            fw_csv.writerow(sglPred)
                for i in range(nnClassCount):
                    chkTotal += 1
                    if sglTruth[i]<=0.5:
                        if sglPred[i]<=0.5:
                            # True Negative
                            TNVal[i]  += 1
                            accVal[i]  += 1
                            accTotal   += 1
                            confVal[i] += 1-sglPred[i]
                            confTotal  += 1-sglPred[i]
                        else:
                            # False Positive
                            FPVal[i]  += 1
                            failVal[i] += 1
                            failTotal  += 1
                            weakVal[i] += 1-sglPred[i]
                            weakTotal  += 1-sglPred[i]
                    else:
                        if sglPred[i]>=0.5:
                            # True Positive
                            TPVal[i]  += 1
                            accVal[i]  += 1
                            accTotal   += 1
                            confVal[i] += sglPred[i]
                            confTotal  += sglPred[i]
                        else:
                            # Flase Negative
                            FNVal[i]  += 1
                            failVal[i] += 1
                            failTotal  += 1
                            weakVal[i] += sglPred[i]
                            weakTotal  += sglPred[i]
                        
            if nBatch==0:
                outTruth = sglTruth
                outPred = sglPred
            else:
                outTruth = np.vstack((outTruth,sglTruth))
                outPred  = np.vstack((outPred, sglPred))
            if nBatch%dataStep==0:
                if debug > 1: fw_csv.writerow((epoch, nBatch, lv))
            nBatch += 1
            if nBatch > maxStep:
                break
               
        bar.finish()
        accTotal  = accTotal/chkTotal
        failTotal = failTotal/chkTotal
        confTotal = confTotal/chkTotal
        weakTotal = weakTotal/chkTotal
        writer.add_scalar('Train/Accuracy',   accTotal,  epoch)
        writer.add_scalar('Train/Failure',    failTotal, epoch)
        writer.add_scalar('Train/Confidence', confTotal, epoch)
        writer.add_scalar('Train/Weakness',   weakTotal, epoch)
        avgLossT = lossVal/nBatch
        lossTrain.append(avgLossT)
        print('Average loss: %0.5f '%(avgLossT),end='')
        #evaluate model improvements for this epoch
        model.eval()
        nBatch=0
        lossVal = 0
        print(" Evaluating")
        
        maxStep=maxEval
        bar = progressbar.ProgressBar(max_value=maxStep)

        with torch.no_grad():
            for img_path, images, labels  in (dataLoaderVal):
                bar.update(nBatch)
                #print(".",end='')
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs = outputs.to(device)
                loss = criterion(torch.sigmoid(outputs),labels)
                lv = loss.item()
                lossVal += lv
                nBatch += 1
                if nBatch > maxStep:
                    break
        
        bar.finish()
        
        avgLossE = lossVal/nBatch
        lossEval.append(avgLossE)
        if debug > 1: fw_csv.writerow((epoch, avgLossT, avgLossE))
        print('Eval avg. loss - %0.5f '%(avgLossE))
        print("Saving model")
        torch.save(model.state_dict(), checkpoint)

    if debug > 1: f_csv.close()

    return model, lossTrain, lossEval
    
def test_model(model, checkpoint, class_names, maxBatch=50):
    cudnn.benchmark = True
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    if debug > 1: 
        f_csv = open('TestOutput'+timestampLaunch+'.csv','w', newline='')
        fw_csv = csv.writer(f_csv)

    if checkpoint != None and use_gpu:
        model.load_state_dict(torch.load(checkpoint))
        model = model.to(device)

    model.eval()
    nBatch=0
    lossVal = 0
    chkTotal = 0
    accVal = [0 for i in range(len(class_names))]
    failVal = [0 for i in range(len(class_names))]
    confVal = [0 for i in range(len(class_names))]
    weakVal = [0 for i in range(len(class_names))]
    accTotal = 0
    failTotal = 0
    confTotal = 0
    weakTotal = 0
    TPVal = [0 for i in range(len(class_names))]
    TNVal = [0 for i in range(len(class_names))]
    FPVal = [0 for i in range(len(class_names))]
    FNVal = [0 for i in range(len(class_names))]
    outTruth = []
    outPred  = []
    print(" Testing")
    maxStep = maxBatch
    bar = progressbar.ProgressBar(max_value=maxStep)
    maxPred = 0
    with torch.no_grad():
        for img_path, images, labels  in (dataLoaderTest):
            bar.update(nBatch)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.to(device)

            sglTruth = np.array(labels[0].cpu(), dtype='int')
            sglPred  = np.array(torch.sigmoid(outputs[0]).cpu(), dtype='float')
            if debug > 1: print("Truth: ",sglTruth)
            if debug > 1: print("Predi: ",sglPred)
#            fw_csv.writerow(sglPred)
            for i in range(len(class_names)):
                if sglPred[i] > maxPred: maxPred = sglPred[i]
                chkTotal += 1
                if sglTruth[i]<=0.5:
                    if sglPred[i]<=0.5:
                        # True Negative
                        TNVal[i]   += 1
                        accVal[i]  += 1
                        accTotal   += 1
                        confVal[i] += 1-sglPred[i]
                        confTotal  += 1-sglPred[i]
                    else:
                        # False Positive
                        FPVal[i]   += 1
                        failVal[i] += 1
                        failTotal  += 1
                        weakVal[i] += sglPred[i]
                        weakTotal  += sglPred[i]
                else:
                    if sglPred[i]>=0.5:
                        # True Positive
                        TPVal[i]   += 1
                        accVal[i]  += 1
                        accTotal   += 1
                        confVal[i] += sglPred[i]
                        confTotal  += sglPred[i]
                    else:
                        # Flase Negative
                        FNVal[i]   += 1
                        failVal[i] += 1
                        failTotal  += 1
                        weakVal[i] += 1-sglPred[i]
                        weakTotal  += 1-sglPred[i]
                        
            if nBatch==0:
                outTruth = sglTruth
                outPred = sglPred
            else:
                outTruth = np.vstack((outTruth,sglTruth))
                outPred  = np.vstack((outPred, sglPred))
            roc_auc = 0 # tbd
            nBatch += 1
            if nBatch > maxBatch:
                nBatch -= 1
                break

    bar.finish()
    print(" maximum prediction: ", maxPred)
    print("\n==============================================================")
    if nBatch > 0:
        if chkTotal > 0:
            accTotal = accTotal/chkTotal
            failTotal = failTotal/chkTotal
            confTotal = confTotal/chkTotal
            weakTotal = weakTotal/chkTotal
        if debug > 1: fw_csv.writerow(("Accuracy", accTotal, failTotal, confTotal, weakTotal))
        print("Accuracy: \t",accTotal,"Fail: \t", failTotal,"Conf: \t", confTotal,"Weak: \t", weakTotal,"Batch#: \t", nBatch)
        for i in range(len(class_names)):
            if accVal[i] > 0:
                confVal[i] = confVal[i]/accVal[i]
            else:
                confVal[i] = 0
            if failVal[i] > 0:
                weakVal[i] = weakVal[i]/failVal[i]
            else:
                weakVal[i] = 0
            accVal[i] = accVal[i]/nBatch
            failVal[i] = failVal[i]/nBatch
            if debug > 1: fw_csv.writerow((class_names[i], accVal[i], failVal[i], confVal[i], weakVal[i], TPVal[i], TNVal[i], FPVal[i], FNVal[i] ))
            print(class_names[i],"\t",accVal[i],"\t",failVal[i],"\t", confVal[i],"\t", weakVal[i])

    if debug > 1: f_csv.close()

    return outTruth, outPred
    
if doTrain:
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    model, lTrain, lEval = train_model(model, trCheckPoint, criterion, optimizer, num_epochs=trMaxEpoch, maxBatch=trMaxBatch, maxEval=trMaxEval)
    batch = [i for i in range(len(lTrain))]
    plt.plot(batch, lTrain, label = "train")
    plt.plot(batch, lEval, label = "eval")
    plt.xlabel("Nb of batches")
    plt.ylabel("BCE loss")
    plt.title(trNetName+' BCE loss evolution')
    plt.legend()
    plt.savefig(trNetName+'_BCEloss_'+timestampLaunch+'.png', dpi=600)
    #plt.pause(5)  
    #plt.show()

model.eval()

#debug limit batch runs
if trBatchLimit>0: 
    trMaxBatch = trBatchLimit
else:
    trMaxBatch = len(datasetTest)
if doTest:
    truth, predict = test_model(model, trCheckPoint, nnClassNames, maxBatch=trMaxBatch)

    aucAvg = 0 
    aucAvgW = 0
    aucAvgN = 0
    fig= plt.figure(figsize=(19,9.5))
    plt.title(trNetName)

    for i in range(nnClassCount):
        fpr, tpr, thresholds = metrics.roc_curve(truth[:,i], predict[:,i], pos_label=1)
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampLaunch = timestampDate + '-' + timestampTime
        if debug > 1: 
            f_csv = open('ROC'+nnClassNames[i]+timestampLaunch+'.csv','w', newline='')
            fw_csv = csv.writer(f_csv)
            fw_csv.writerow(fpr)
            fw_csv.writerow(tpr)
            f_csv.close()
        roc_auc = metrics.auc(fpr, tpr)
        aucAvg += roc_auc
        aucAvgN += len(thresholds)
        aucAvgW += roc_auc*len(thresholds)
        print(nnClassNames[i], roc_auc, len(thresholds))
        f = plt.subplot(2, 8, i+1)
        plt.title(nnClassNames[i])

        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    aucAvg = aucAvg/nnClassCount
    aucAvgW = aucAvgW/aucAvgN
    print("Average AUROC : ", aucAvg)
    print("Weighted Average AUROC : ", aucAvgW)
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    plt.savefig(trNetName+'_AuRoc_'+timestampLaunch+'.png', dpi=600)
    #if doShow: plt.pause(5)  
    if doShow: plt.show()
    
    
    
writer.close()



