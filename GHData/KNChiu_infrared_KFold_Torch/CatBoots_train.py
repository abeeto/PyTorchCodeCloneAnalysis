#%%
from functools import total_ordering
import os
from unittest.mock import patch
import cv2
import random
import io

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix ,roc_curve, auc, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt 
from itertools import cycle
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# from model.patch_convmix_convnext import PatchConvmixConvnext
# from model.patch_RepLKNet_DRSN import PatchRepLKNetDRSN
from model.patch_convmix_Attention import PatchConvMixerAttention

from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

import catboost as cb

from model.load_dataset import MyDataset, MultiEpochsDataLoader, CudaDataLoader
from model.assessment_tool import MyEstimator

import math
import logging

from argparse import ArgumentParser

SEED = 42
if SEED:
    '''設定隨機種子碼'''
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


def test_model(model, test_loader, classes):
    # test
    if SAVEBAST: 
        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_bast.pth"
    else:
        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"

    model.eval()
    model.load_state_dict(torch.load(saveModelpath))
    model.to(device)
    # correct = 0
    y_true = []
    y_pred = []
    y_pred_score = []
    keyLabel = []

    with torch.no_grad():
        for idx, (x, y, key) in enumerate(test_loader):
            keyLabel += key
            pred = model(x.to(device))
            pred, gap, test_feature = pred

            y_pred_score += pred.tolist()

            # 計算是否正確
            pred = torch.max(pred.data, 1)[1] 
            # correct += (pred == y)).sum()

            y_true += y.tolist()
            y_pred += pred.tolist()
            
        
        Accuracy, Specificity, Sensitivity, error_list, _ = MyEstimator.confusion(y_true, y_pred, val_keyLabel=keyLabel, logPath = None, classes = classes)
        roc_auc, _ = MyEstimator.compute_auc(y_true, y_pred_score, classes)

        return Accuracy, roc_auc, Specificity, Sensitivity, y_true, y_pred, y_pred_score, gap, keyLabel, error_list

def load_feature(dataloader, model):
    if SAVEBAST: 
        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_bast.pth"
    else:
        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"

    model.eval()
    model.load_state_dict(torch.load(saveModelpath))
    model.to(device)

    feature, label, keyLabel = [], [], []
    for idx, (x, y, key) in enumerate(dataloader):
        keyLabel += key
        _, _, featureOut = model(x.to(device))
        
        featureOut = featureOut[0].to('cpu').detach().numpy()
        featureOut = featureOut.reshape((1, -1))[0]

        feature.append(featureOut)
        label.append(y.to('cpu').detach().numpy())
        
    feature = np.array(feature)
    label = np.array(label)
    return feature, label, keyLabel

def catboots_fit(train_data, train_label, val_data, val_label, iterations):
    # cbc = cb.CatBoostClassifier(random_state=SEED, use_best_model=True, iterations=iterations, depth = CatBoost_depth,random_seed=SEED)
    # cbc = cb.CatBoostClassifier(iterations=10000,learning_rate=0.1,max_depth=7,verbose=100,
    #                                   early_stopping_rounds=500,task_type='GPU',eval_metric='AUC',random_seed=SEED)
    cbc = cb.CatBoostClassifier(
                               loss_function='MultiClass',
                                eval_metric='WKappa',
                               task_type="GPU",
                            #    learning_rate=0.01,
                               iterations=iterations,
                               od_type="Iter",
                                depth=2,
                               early_stopping_rounds=100,
                                #l2_leaf_reg=10,
                                #border_count=96,
                               random_seed=42,
                                use_best_model=True
                              )

    cbc.fit(train_data, train_label,
            eval_set = [(val_data, val_label)],
            verbose=False,
            plot=False
            )
    predict = cbc.predict(val_data)
    predict_Probability = cbc.predict(val_data, prediction_type='Probability')
    return predict, predict_Probability

def XGBoost_fit(train_data, train_label, val_data, val_label, iterations):
    xgbc = XGBClassifier(n_estimators=1000, 
                            max_depth=6, 
                            learning_rate=0.05,
                            objective='binary:logistic'
                            )

    xgbc.fit(train_data, train_label,
            eval_set = [(val_data, val_label)],
            verbose=False)

    predict = xgbc.predict(val_data)
    predictions = xgbc.predict_proba(val_data)

    return predict, predictions

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

if __name__ == '__main__':
    parser = ArgumentParser()                                   # 使用超參數套件
    parser.add_argument('--train_mode', default=0, type=int)    # 使用參數控制訓練模式
    args = parser.parse_args()                                  # 解析
    train_mode = args.train_mode                                # 指派訓練模式

    if train_mode == 0:
        modelName = "7d2G-3d2GB_1GB_X4_CA_SA_paper_"
    elif train_mode == 1:
        modelName = "7d4G-5d3GB_1GB_X4_CA_SA_paper_"
    elif train_mode == 2:
        modelName = "7d3G-5d2GB_1GB_X4_CA_SA_paper_"
    elif train_mode == 3:
        modelName = "7d1G-3d1GB_1GB_X5_paper_"


    CLASSNANE = ['Infect', 'Ischemia']

    SAVEPTH = True
    SAVEIDX = True
    RUNML = True
    SAVEBAST = False
    # WANDBRUN = True
    WANDBRUN = False

    CNN_DETPH = 3
    KERNELSIZE = 7
    
    WARMUP_ITER = 50
    # WARMUP_ITER = 100

    # KFOLD_N = 2
    # EPOCH = 1
    KFOLD_N = 10
    EPOCH = 683
    TRYMODEL = False
    VRAM_FAST = False

    BATCHSIZE = 16
    LR = 0.01
    # LR = 0.0001
    DRAWIMG = 50

    CATBOOTS_INTER = 1000

    LOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
    DATAPATH = r'C:\Data\surgical_temperature\color\via_gray\\'
    WANDBDIR = r'C:\Data\surgical_temperature\trainingLogs\\'


    MyEstimator = MyEstimator()
    Dataload = MyDataset(DATAPATH, LOGPATH, 2)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    start = time.time()
    # 建立 log
    # logPath = LOGPATH + "//logs//" + str(time.strftime("%m%d_%H%M%S", time.localtime()))
    logPath = LOGPATH + "//logs//" + str(modelName)
    if not os.path.isdir(logPath):
        os.mkdir(logPath)
        os.mkdir(logPath+'//img//')

    logger = get_logger(logPath + '//XGBtest.log', name=str(modelName))     # 建立 logger
    

    logger.info("================================= CNN -> ML ============================================")
    # dataset = ImageFolder(DATAPATH, transform)          # 輸入數據集
    dataset  = Dataload
    kf = KFold(n_splits = KFOLD_N, shuffle = True)
    Kfold_cnt = 0

    total_true = []
    total_pred = []
    total_pred_score = []

    ML_total_true = []
    ML_total_pred = []
    ML_total_pred_score = []
    total_keyLabel = []
    ML_total_keyLabel = []

    # KFOLD
    for train_idx, val_idx in kf.split(dataset):
        Kfold_cnt += 1

        if WANDBRUN:
            wb_run = wandb.init(project='infraredThermal_kfold', reinit=True, group="ForPaper", name=str(str(modelName)+"_K="+str(Kfold_cnt)), dir = WANDBDIR)
        
        if SAVEIDX:
            with open(logPath + '//'+ 'kfold_idx.json','a+',encoding="utf-8") as json_file:
                json_file.seek(0)  
                if json_file.read() =='':  
                    data = {}
                else:
                    json_file.seek(0)
                    data = json.load(json_file)

                data['Kfold_cnt' + str(Kfold_cnt)] = {'train_idx':train_idx.tolist(), 'val_idx':val_idx.tolist()}

                json_file.seek(0)
                json_file.truncate()
                json.dump(data, json_file, indent=2, ensure_ascii=False)
        
        # 重組 kfold 數據集
        train = Subset(dataset, train_idx)
        val = Subset(dataset, val_idx)

        if VRAM_FAST:
            train_loader = MultiEpochsDataLoader(train, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)    # 使用客製化加速載入訓練集
            val_loader = MultiEpochsDataLoader(val, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)

            train_loader = CudaDataLoader(train_loader, device)   # 放入vram加速
            val_loader = CudaDataLoader(val_loader, device)

        else:
            train_loader = DataLoader(train, shuffle = np.True_, batch_size=BATCHSIZE, num_workers = 1, persistent_workers = True)
            val_loader = DataLoader(val, shuffle = True, batch_size=BATCHSIZE, num_workers = 1, persistent_workers = True)


        # 匯入模型
        model = PatchConvMixerAttention(dim = 768, depth = CNN_DETPH, kernel_size = KERNELSIZE, patch_size = 16, n_classes = len(CLASSNANE), train_mode = train_mode).to(device)
        
        # Train
        # fit_model(model, train_loader, val_loader, CLASSNANE)

        # Test
        Accuracy, roc_auc, Specificity, Sensitivity, kfold_true, kfold_pred, kfold_pred_score, gap, val_keyLabel, error_list = test_model(model, val_loader, CLASSNANE)

        total_true += kfold_true
        total_pred += kfold_pred
        total_pred_score += kfold_pred_score
        total_keyLabel += val_keyLabel

        if roc_auc != -1:
            roc_auc = max(roc_auc.values())

        # print("==================================== CNN Training=================================================")
        # print('Kfold : {} , Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(Kfold_cnt, Accuracy, roc_auc, Specificity, Sensitivity))
        # print("True : 1 but 0 :")
        # print(error_list['1_to_0'])
        # print("True : 0 but 1 :")
        # print(error_list['0_to_1'])
        # print("===================================================================================================")
    

        if WANDBRUN:
            wb_run.log({
                        "CNN Accuracy" : Accuracy,
                        "CNN AUC" : roc_auc,
                        "CNN Specificity" : Specificity,
                        "CNN Sensitivity" : Sensitivity
                        })

# ML ===============================================================
        if RUNML:
            # 強分類器
            # 提取特徵圖
            # print("================================= ML Training ===============================================")
            ML_train_loader = DataLoader(train, shuffle = np.True_, num_workers = 1, persistent_workers = True)
            ML_val_loader = DataLoader(val, shuffle = True, num_workers = 1, persistent_workers = True)

            feature_train_data, feature_train_label, train_keyLabel = load_feature(ML_train_loader, model)
            feature_val_data, feature_val_label, ML_val_keyLabel = load_feature(ML_val_loader, model)
            ML_total_keyLabel += ML_val_keyLabel

            # predict, predict_Probability = catboots_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER)
            predict, predict_Probability = XGBoost_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER)
            ML_roc_auc, compute_img = MyEstimator.compute_auc(feature_val_label, predict_Probability, CLASSNANE, logPath+"\\img", mode = 'ML_' + str(Kfold_cnt))
            ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, confusion_img = MyEstimator.confusion(feature_val_label, predict, ML_val_keyLabel, classes = CLASSNANE, logPath = logPath+"\\img", mode ='ML_' + str(Kfold_cnt))

            if ML_roc_auc != -1:
                ML_roc_auc = max(ML_roc_auc.values())
            
            logger.info("Kfold = [{}]\t".format(Kfold_cnt))
            logger.info("Accuracy    : {:.2} => {:.2}\t  AUC         : {:.2} => {:.2}".format(Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
            logger.info("Specificity : {:.2} => {:.2}\t  Sensitivity : {:.2} => {:.2}".format(Specificity, ML_Specificity, Sensitivity, ML_Sensitivity))
            logger.info("-------------------------------------------------------------------------------------")

            if WANDBRUN:
                wb_run.log({
                            "ML Accuracy" : ML_Accuracy,
                            "ML AUC" : ML_roc_auc,
                            "ML Specificity" : ML_Specificity,
                            "ML Sensitivity" : ML_Sensitivity
                            })

            ML_total_true += feature_val_label.tolist()
            ML_total_pred += predict.tolist()
            ML_total_pred_score += predict_Probability.tolist()
        
        if TRYMODEL:
            ML_total_true = feature_val_label
            ML_total_pred = predict
            ML_total_pred_score = predict_Probability
            break

# Kflod end ================================================
    # Kfold CNN 結束交叉驗證

    Accuracy, Specificity, Sensitivity, error_list, confusion_img = MyEstimator.confusion(total_true, total_pred, total_keyLabel, classes = CLASSNANE, logPath = logPath, mode = 'Kfold_CNN')
    roc_auc, compute_img = MyEstimator.compute_auc(total_true, total_pred_score, CLASSNANE, logPath, mode = 'Kfold_CNN')
    
    # print("==================================== CNN Training=================================================")
    # print("True : 1 but 0 :")        # print(error_list['1_to_0'])
    # print("True : 0 but 1 :")
    # print(error_list['0_to_1'])


    if roc_auc != -1:
            roc_auc = float(max(roc_auc.values()))
    if WANDBRUN:
        wb_run.log({
                    "KFold_CNN_ML Accuracy" : Accuracy,
                    "KFold_CNN_ML AUC" : roc_auc,
                    "KFold_CNN_ML Specificity" : Specificity.item(),
                    "KFold_CNN_ML Sensitivity" : Sensitivity.item(),
                    "KFold_CNN_ML compute": [wandb.Image(compute_img)],
                    "KFold_CNN_ML confusion": [wandb.Image(confusion_img)]
                    })

    if RUNML:
        # Kfold ML 結束交叉驗證
        ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, compute_img = MyEstimator.confusion(ML_total_true, ML_total_pred, ML_total_keyLabel, classes = CLASSNANE, logPath = logPath, mode = 'Kfold_ML')
        ML_roc_auc, confusion_img = MyEstimator.compute_auc(ML_total_true, ML_total_pred_score, CLASSNANE, logPath, mode = 'Kfold_ML')

        # print("==================================== ML Training=================================================")
        

        if ML_roc_auc != -1:
            ML_roc_auc = float(max(ML_roc_auc.values()))

        
        logger.info("=============================== KFlod Finish =====================================================")
        logger.info("Total Kfold = [{}]\t".format(KFOLD_N))
        logger.info("Accuracy    : {:.2} => {:.2}\t  AUC         : {:.2} => {:.2}".format(Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
        logger.info("Specificity : {:.2} => {:.2}\t  Sensitivity : {:.2} => {:.2}".format(Specificity.item(), ML_Specificity.item(), Sensitivity.item(), ML_Sensitivity.item()))
        logger.info("===================================================================================================")
        logger.info("KFlod time : " + str(time.time() - start) + " s")
        logger.info("True : 1 but 0 :")
        logger.info(error_list['1_to_0'])
        logger.info("True : 0 but 1 :")
        logger.info(error_list['0_to_1'])
        
        logger.info("Model : \n" + str(model))

        if WANDBRUN:
            wb_run.log({
                        "KFold_CNN_ML Accuracy" : ML_Accuracy,
                        "KFold_CNN_ML AUC" : ML_roc_auc,
                        "KFold_CNN_ML Specificity" : ML_Specificity.item(),
                        "KFold_CNN_ML Sensitivity" : ML_Sensitivity.item(),
                        "KFold_CNN_ML compute": [wandb.Image(compute_img)],
                        "KFold_CNN_ML confusion": [wandb.Image(confusion_img)]
                    })

    torch.cuda.empty_cache()    # 釋放記憶體
    logging.shutdown()          # 關閉logger


    if WANDBRUN:
        wb_run.finish()
