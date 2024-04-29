import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import networks_v1 as networks
import sys
import module
import gAtt
from data import speechDataset, collate_fn, att2img, plotAtt, plotMel
import numpy as np
import os
from tqdm import tqdm
from params import param
import shutil

class graph:
    def __init__(self, trNet):
        self.trNet = trNet
        if self.trNet is "t2m":
            self.trainGraph = networks.t2mGraph().to(DEVICE)
            
        elif self.trNet is "SSRN":
            self.trainGraph = networks.SSRNGraph().to(DEVICE)

# if load
def load(graph, logDir, optimizer, networkPath):
    log = np.genfromtxt(logDir, delimiter=',')
    logList = list(log)
    globalStep = int(logList[0][0])
    newMPath = os.path.join(networkPath, 'best_{}'.format(globalStep))
    ckptPath = os.path.join(newMPath, 'bestModel_{}.pth'.format(globalStep))
    ckpt = torch.load(ckptPath)
    graph.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    globalStep += 1

    return globalStep, logList

def train(graph, trNet, trainLoader, valLoader, networkPath, writer):
# 1. Load Data from pre-processed wav and scripts
# 2. make batches
# 3. build Graph
# 4. forward
# 5. train
    optimizer = torch.optim.Adam(graph.parameters(), lr=param.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.93)
    loss10klist = []
    if os.path.exists(os.path.join(networkPath, 'log.csv')):
        logDir = os.path.join(networkPath, 'log.csv')
        globalStep, lossList = load(graph, logDir, optimizer, networkPath)
    else:
        lossList = [(99999, 9999.)]
        os.mkdir(os.path.join(networkPath, 'best_{}'.format(lossList[0][0])))
        globalStep = 0
    
    lossDic = {'lossGstep':0., 'L1lossGstep':0., 'BinlossGstep':0., 'AttlossGstep':0.}
    lossL1 = torch.nn.L1Loss().to(DEVICE)
    lossBin = torch.nn.BCELoss().to(DEVICE)

    while globalStep < param.maxStep:
        for key in lossDic.keys():
            lossDic[key] = 0.
        for _, batchData in tqdm(enumerate(trainLoader), leave=False, unit='B'):
            optimizer.zero_grad()
            # calculate loss value
            if trNet is "t2m":
                batchText, batchMel, _, batchgMat, _ = batchData
                predFirst = torch.zeros(len(batchText), 1, param.n_mels).to(DEVICE) # for shift
                batchText, batchMel = batchText.to(DEVICE), batchMel.to(DEVICE) # (B, N) (B, T, n_mels)
                inputMel = torch.cat((predFirst, batchMel[:, :-1, :]), 1)
                predMel, Att, _ = graph(batchText, inputMel) # prediction results, attention matrix
        
                predMel = predMel.transpose(1,2)
                l1Loss = lossL1(predMel, batchMel)
                BinLoss = lossBin(predMel, batchMel)
                attLoss = gAtt.gAttlossNT(Att, batchgMat, DEVICE)
                loss = attLoss + l1Loss + attLoss
                lossDic['AttlossGstep'] += attLoss.item()

            elif trNet is "SSRN":
                _, batchMel, batchMag, _ = batchData
                batchMel, batchMag = batchMel.to(DEVICE), batchMag.to(DEVICE)
                predMag = graph(batchMel)
                predMag = predMag.transpose(1, 2)
                
                l1Loss = lossL1(predMag, batchMag)
                BinLoss = lossBin(predMag, batchMag)
                loss = l1Loss + BinLoss
            
            lossDic['lossGstep'] += loss.item()
            lossDic['L1lossGstep'] += l1Loss.item()
            lossDic['BinlossGstep'] += BinLoss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(graph.parameters(), 2.0)
            optimizer.step()
            # scheduler.step()
            
            if globalStep % 10000 == 0:
                loss10klist = saveModel10k(globalStep, graph, optimizer,
                                            loss, networkPath, loss10klist)
                np.savetxt(os.path.join(networkPath, 'log10k.csv'),loss10klist , delimiter=',')
            globalStep += 1

        for key in lossDic.keys():
            lossDic[key] /= len(trainLoader)
    
        writer.add_scalar('train/train_loss', lossDic['lossGstep'], global_step=globalStep)
        writer.add_scalar('train/L1_loss', lossDic['L1lossGstep'], global_step=globalStep)
        writer.add_scalar('train/Bin_loss', lossDic['BinlossGstep'], global_step=globalStep)
        writer.add_scalar('train/Att_loss', lossDic['AttlossGstep'], global_step=globalStep)

        print('[TRAIN] loss at {} globalStep : {}'.format(globalStep, lossDic['lossGstep']))
        if globalStep % 1 == 0:
            graph.eval()
            # validate

            if trNet is "t2m":
                # save attention matrix image
                valLoss, lossList, valAtt = validate(graph, trNet, networkPath, valLoader, optimizer, 
                                                # scheduler,
                                                 lossL1, globalStep, lossList)
                writer.add_scalar('train/validation_loss', valLoss, global_step=globalStep)
                writer.add_image('Attention Matrix', att2img(valAtt), global_step=globalStep)
            elif trNet is "SSRN":
                valLoss, lossList = validate(graph, trNet, networkPath, valLoader, optimizer, 
                                             # scheduler,
                                             lossL1, globalStep, lossList)
                writer.add_scalar('val/validation_loss', valLoss, global_step=globalStep)
            graph.train()

def validate(graph, trNet, networkPath, valLoader, optimizer, 
            #  scheduler, 
             lossL1, globalStep, lossList):
    with torch.no_grad():
        valLoss = 0.
        for _, batchData in enumerate(valLoader):
            if trNet is "t2m":
                batchText, batchMel, _, _, _ = batchData
                predFirst = torch.zeros(len(batchText), 1, param.n_mels).to(DEVICE)
                batchText, batchMel = batchText.to(DEVICE), batchMel.to(DEVICE)
                inputMel = torch.cat((predFirst, batchMel[:, :-1, :]), 1)
                predMel, Att, _ = graph(batchText, inputMel, False) # prediction results, attention matrix
                
                predMel = predMel.transpose(1,2)
                batchLoss = lossL1(predMel, batchMel)
                valLoss += batchLoss.item()

            elif trNet is "SSRN":
                _, batchMel, batchMag, _ = batchData
                batchMel, batchMag = batchMel.to(DEVICE), batchMag.to(DEVICE)
                predMag = graph(batchMel)
                predMag = predMag.transpose(1, 2)

                batchLoss = lossL1(predMag, batchMag)
                valLoss += batchLoss.item()
                

        valLoss = valLoss / len(valLoader) 
        if lossList[-1][-1] > valLoss:
            prevLoss = lossList[-1][-1]
            lossList = saveModel(globalStep, graph, optimizer,
                                #  scheduler, 
                                 valLoss, networkPath, lossList)
            print('[VAL] loss at {} globalStep. Prev : {} , New : {} MODEL SAVED'.format(globalStep, prevLoss, valLoss))
            np.savetxt(os.path.join(networkPath, 'log.csv'),lossList , delimiter=',')
        if trNet is "t2m":
            plotAtt(att2img(Att[5]), batchText[5], globalStep, networkPath)
            plotMel(predMel[5], globalStep, networkPath)
            return valLoss, lossList, Att[5]
        elif trNet is "SSRN":
            return valLoss, lossList

def saveModel(globalStep, graph, optimizer, 
            #   scheduler, 
              loss, networkPath, lossList):
    # with the best 5 good model [(globalStep, loss)]
    # When current loss is lower than lowest loss valuen in the list.
    # save state dict
    #

    if len(lossList) > 4:
        # delete directory of a model with highest loss 
        shutil.rmtree(os.path.join(networkPath, 'best_{}'.format(int(lossList[-1][0]))))
        lossList[-1] = (globalStep, loss)
    else:
        lossList.append((globalStep, loss))

    newMPath = os.path.join(networkPath, 'best_{}'.format(globalStep))
    if not os.path.exists(newMPath): 
        os.mkdir(newMPath)

    torch.save({
        'globalStep': globalStep,
        'model_state_dict': graph.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, os.path.join(newMPath, 'bestModel_{}.pth'.format(globalStep)))
    
    lossList.sort(key = lambda x: x[-1])
    
    return lossList

def saveModel10k(globalStep, graph, optimizer, loss, networkPath, lossList):

    lossList.append((globalStep, loss))
    newMPath = os.path.join(networkPath, 'best_{}'.format(globalStep))
    if not os.path.exists(newMPath): 
        os.mkdir(newMPath)

    torch.save({
        'globalStep': globalStep,
        'model_state_dict': graph.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, os.path.join(newMPath, 'bestModel_{}.pth'.format(globalStep)))
    
    return lossList

# dropout layer
# normalization

if __name__ == "__main__":
    
    try:
        trNet = int(sys.argv[1])
    except:
        trNet = 0
    try:
        retrain = int(sys.argv[3])
    except:
        retrain = 0 # no re-train pre-trained model
    try :
        modelIdx = int(sys.argv[2])
        print('Index of model directory starts from {} : model_[Idx]'.format(modelIdx))
    except:
        print('Index of model directory starts from 0 : model_[Idx]')
        modelIdx = 0

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choosing Dataset
    if trNet is 0:
        trNet = "t2m"
        g = graph(trNet).trainGraph
        trainDataset = speechDataset(trNet, 0)
        valDataset = speechDataset(trNet, 1)
    elif trNet is 1:
        trNet = "SSRN"
        g = graph(trNet).trainGraph
        
        trainDataset = speechDataset(trNet, 0)
        valDataset = speechDataset(trNet, 1)
    else:
        raise ValueError("Please Use proper argv, 0: t2m, 1:SSRN, your input : {}".format(trNet))
    print('Train and Validation Data Loaded successfully.')
    if not os.path.exists(os.path.abspath('../DCTTS.results')):
        os.mkdir(os.path.abspath('../DCTTS.results'))
    while 1:
        modelDir = os.path.abspath('../DCTTS.results/model_{}'.format(modelIdx))
        networkPath = os.path.abspath(os.path.join(modelDir, trNet))
        if retrain:
            break
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)
            os.mkdir(networkPath)
            break
        else:
            if not os.path.exists(networkPath):
                os.mkdir(networkPath)
                break
            else:
                modelIdx += 1
    
    print('Directory of Model : {}'.format(modelDir))
    print('Directory of Network : {}'.format(networkPath))
    print('{} network will be training...'.format(trNet))
    writer = SummaryWriter(networkPath)

    trainLoader = DataLoader(dataset=trainDataset,
                             batch_size=param.B,
                             shuffle=True,
                             collate_fn=collate_fn,
                             drop_last=True)
    valLoader = DataLoader(dataset=valDataset,
                            batch_size=param.B,
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=True)
    train(g, trNet, trainLoader, valLoader, networkPath, writer)