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
from sklearn.metrics import confusion_matrix ,roc_curve, auc, accuracy_score
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


import torch.nn.functional as F 

def linear_combination(x, y, epsilon):  
    return epsilon*x + (1-epsilon)*y
 
def reduce_loss(loss, reduction='mean'): 
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 
 
class LabelSmoothingCrossEntropy(torch.nn.Module): 
    def __init__(self, epsilon:float=0.1, reduction='mean'): 
        super().__init__() 
        self.epsilon = epsilon 
        self.reduction = reduction 
 
    def forward(self, preds, target): 
        n = preds.size()[-1] 
        log_preds = F.log_softmax(preds, dim=-1) 
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) 
        nll = F.nll_loss(log_preds, target, reduction=self.reduction) 
        return linear_combination(loss/n, nll, self.epsilon)


def fit_model(model, train_loader, val_loader, classes):
    # optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    optimizer = torch.optim.SGD(model.parameters(), lr = LR, weight_decay=1e-4)
    # loss_func = FocalLoss(class_num=3, alpha = torch.tensor([0.36, 0.56, 0.72]).to(device), gamma = 4)
    # loss_func = FocalLoss(class_num=len(classes), alpha = torch.tensor([0.44, 0.56]).to(device), gamma = 2)
    loss_func = FocalLoss(class_num=len(classes), alpha = None, gamma = 1)
    # loss_func = LabelSmoothingCrossEntropy() 

    # loss_func = torch.nn.CrossEntropyLoss()

    cos_restart_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)    # (1 + T_mult + T_mult**2) * T_0 // 5,15,35,75,155
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=0, cycle_mult=2.0, max_lr=LR, min_lr=0, warmup_steps=0, gamma=1.0)

    warm_up_iter = WARMUP_ITER
    T_max = WARMUP_ITER	# 周期

    # warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_iter if epoch <= warm_up_iter else 0.5 * ( math.cos((epoch - warm_up_iter) /(T_max - warm_up_iter) * math.pi) + 1)
    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_iter 
    warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)



    mini_val_loss = 100
    for epoch in range(EPOCH):
        training_loss = 0

        train_y_pred_score =[]
        train_y_true = []
        train_y_pred = []
        for idx, (x, y, _) in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()

            output = model(x.to(device))
            outPred = output[0]

            loss = loss_func(outPred, y.to(device))
            training_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            train_y_pred_score += outPred.tolist()

            # 計算是否正確
            pred = torch.max(outPred.data, 1)[1] 

            train_y_true += y.tolist()
            train_y_pred += pred.tolist()

        if epoch+1 < WARMUP_ITER:
            warm_up_scheduler.step()        # WarmUp
        else:
            cos_restart_scheduler.step()    # cos退火

        cur_lr = optimizer.param_groups[-1]['lr']  

        train_roc_auc, _ = MyEstimator.compute_auc(train_y_true, train_y_pred_score, classes)
        train_Accuracy, Specificity, Sensitivity, _ ,_ = MyEstimator.confusion(train_y_true, train_y_pred, val_keyLabel=None, logPath = None, classes = classes)
        
        # val
        model.eval()
        y_pred_score =[]
        y_true = []
        y_pred = []
        with torch.no_grad():
            val_loss = 0
            for idx, (x_, y_, _) in enumerate(val_loader):
                pred, gap, test_feature = model(x_.to(device))
                loss_ = loss_func(pred, y_.to(device))
                val_loss += loss_.item()

                y_pred_score += pred.tolist()
                
                # 計算是否正確
                pred = torch.max(pred.data, 1)[1] 
                
                y_pred += pred.tolist()
                y_true += y_.tolist()

            roc_auc, _ = MyEstimator.compute_auc(y_true, y_pred_score, classes)
            Accuracy, Specificity, Sensitivity, _, _ = MyEstimator.confusion(y_true, y_pred, val_keyLabel=None, logPath = None, classes = classes)

            
        if roc_auc != -1:
            train_roc_auc = max(train_roc_auc.values())
            roc_auc = max(roc_auc.values())

        if WANDBRUN:
            wb_run.log({ 
                        "Epoch" : epoch + 1,
                        "Training Loss": training_loss,
                        "Train ACC" : train_Accuracy,
                        "Train AUC" : train_roc_auc,
                        "Val Loss": val_loss,
                        "Val AUC" : roc_auc,
                        "Val ACC" : Accuracy,
                        "LR" : cur_lr,
                                 # 將可視化上傳 wandb
                    })
            if DRAWIMG != 0:
                if epoch % DRAWIMG == 0 :    
                    if len(gap) > 8:
                        cnt = 8
                    else:
                        cnt = len(gap)

                    plt.close('all')
                    for i in range(cnt):
                        plt.subplot(1, cnt, i+1)
                        plt.imshow(gap[i].cpu().detach().numpy())   # 將注意力圖像取出
                        plt.axis('off')         # 關閉邊框
                    
                    # plt.show()
                    plot_img_np = MyEstimator.get_img_from_fig(plt)    # plt 轉為 numpy
                    plt.close('all')
                    
                    wb_run.log({"val image": [wandb.Image(plot_img_np)]})   # 將可視化上傳 wandb

        if SAVEPTH:
            if SAVEBAST and epoch > 10:
                if mini_val_loss > val_loss:
                    mini_val_loss = val_loss
                    saveModelpath = logPath + "//" + str(Kfold_cnt) + "_bast.pth"
                    torch.save(model.state_dict(), saveModelpath)
            if epoch == EPOCH - 1:
                saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"
                torch.save(model.state_dict(), saveModelpath)

        print('  => Epoch : {}  Training Loss : {:.4e}  Val Loss : {:.4e}  Val ACC : {:.2}  Val AUC : {:.2}'.format(epoch + 1, training_loss, val_loss, Accuracy, roc_auc))
    return training_loss, val_loss 

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
    cbc = cb.CatBoostClassifier(
                               loss_function='MultiClass',
                                eval_metric='WKappa',
                               task_type="GPU",
                               learning_rate=0.01,
                               iterations=iterations,
                               od_type="Iter",
                                depth=6,
                               early_stopping_rounds=100,
                                # l2_leaf_reg=12,
                                # border_count=96,
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
    xgbc = XGBClassifier(n_estimators=iterations, max_depth=12)
    xgbc.fit(train_data, train_label,
            # eval_set = [(val_data, val_label)],
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
    parser = ArgumentParser()                                       # 使用超參數套件
    parser.add_argument('--train_mode', default=0, type=int)        # 使用參數控制訓練模式
    parser.add_argument('--ml_mode', default='XGBoost', type=str)   # 使用參數控制訓練模式
    args = parser.parse_args()                                      # 解析
    train_mode = args.train_mode                                    # 指派訓練模式
    ML_MODE = args.ml_mode                                          # 指派訓練模式

    # ML_MODE = 'XGBoost'
    # ML_MODE = 'CatBoost'



    if train_mode == 0:
        modelName = "(7d4G-5d4GB)_1GB_X4_viajet2_LabelSmooth_wd1e-4"
    elif train_mode == 1:
        modelName = "(7d4G-5d3GB)_1GB_X4_0527_104_FL1_wd1e-4"
    elif train_mode == 2:
        modelName = "(7d3G-5d2GB)_1GB_X4_viajet2_LabelSmooth_wd1e-4"
    elif train_mode == 3:
        modelName = "(7d4G-3d4GB)_1GB_X4_viajet2_LabelSmooth_wd1e-4"
    elif train_mode == 4:
        modelName = "(7d4G-3d3GB)_1GB_X4_viajet2_LabelSmooth_wd1e-4"
    elif train_mode == 5:
        modelName = "(7d3G-3d2GB)_1GB_X4_viajet2_LabelSmooth_wd1e-4"
    elif train_mode == 6:
        modelName = "(9d4G-5d3GB)_1GB_X4_viajet2_LabelSmooth_wd1e-4"
    elif train_mode == 7:
        modelName = "(9d5G-5d4GB)_1GB_X4_dviajet2_LabelSmooth_wd1e-4"


    CLASSNANE = ['Ischemia', 'Infect']

    SAVEPTH = True
    SAVEIDX = True
    RUNML = True
    SAVEBAST = False

    if ML_MODE == 'XGBoost':
        WANDBRUN = True
    elif ML_MODE == 'CatBoost':
        WANDBRUN = False

    CNN_DETPH = 3
    KERNELSIZE = 7
    
    WARMUP_ITER = 50
    # WARMUP_ITER = 100

    # KFOLD_N = 2
    # EPOCH = 1
    KFOLD_N = 10
    EPOCH = 363
    # EPOCH = 683
    TRYMODEL = False
    VRAM_FAST = False

    BATCHSIZE = 16
    LR = 0.01
    # LR = 0.0001
    DRAWIMG = 50

    CATBOOTS_INTER = 1000

    LOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
    DATAPATH = r'C:\Data\surgical_temperature\color\\via_jet2\\'
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
    if ML_MODE == 'XGBoost':
        logger = get_logger(logPath + '//CNN_XGBoost.log', name=str(modelName))     # 建立 logger
    elif ML_MODE == 'CatBoost':
        logger = get_logger(logPath + '//CNN_CatBoost.log', name=str(modelName))

    # logger.info("================================= CNN -> ML ============================================")
    # dataset = ImageFolder(DATAPATH, transform)          # 輸入數據集
    dataset  = Dataload
    kf = KFold(n_splits = KFOLD_N, shuffle = True)
    Kfold_cnt = 0

    total_true = []
    total_pred = []
    total_pred_score = []
    ML_ACC_list = []
    ML_AUC_list = []

    ML_total_true = []
    ML_total_pred = []
    ML_total_pred_score = []
    total_keyLabel = []
    ML_total_keyLabel = []

    # KFOLD
    for train_idx, val_idx in kf.split(dataset):
        Kfold_cnt += 1

        if WANDBRUN:
            wb_run = wandb.init(project='infraredThermal_kfold', reinit=True, group="newdata", name=str(str(modelName)+"_K="+str(Kfold_cnt)), dir = WANDBDIR)
        
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
        
        if ML_MODE == 'XGBoost':
            # Train
            fit_model(model, train_loader, val_loader, CLASSNANE)

        # Test
        Accuracy, roc_auc, Specificity, Sensitivity, kfold_true, kfold_pred, kfold_pred_score, gap, val_keyLabel, error_list = test_model(model, val_loader, CLASSNANE)

        total_true += kfold_true
        total_pred += kfold_pred
        total_pred_score += kfold_pred_score
        total_keyLabel += val_keyLabel

        if roc_auc != -1:
            roc_auc = max(roc_auc.values())

        print("==================================== CNN Training=================================================")
        print('Kfold : {} , Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(Kfold_cnt, Accuracy, roc_auc, Specificity, Sensitivity))
        # print("True : 1 but 0 :")
        # print(error_list['1_to_0'])
        # print("True : 0 but 1 :")
        # print(error_list['0_to_1'])
        print("===================================================================================================")
    

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
            print("================================= ML Training ===============================================")
            ML_train_loader = DataLoader(train, shuffle = np.True_, num_workers = 1, persistent_workers = True)
            ML_val_loader = DataLoader(val, shuffle = True, num_workers = 1, persistent_workers = True)

            feature_train_data, feature_train_label, train_keyLabel = load_feature(ML_train_loader, model)
            feature_val_data, feature_val_label, ML_val_keyLabel = load_feature(ML_val_loader, model)
            ML_total_keyLabel += ML_val_keyLabel

            if ML_MODE == 'XGBoost':
                predict, predict_Probability = XGBoost_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER)
            elif ML_MODE == 'CatBoost':
                predict, predict_Probability = catboots_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER)
            
            ML_roc_auc, compute_img = MyEstimator.compute_auc(feature_val_label, predict_Probability, CLASSNANE, logPath+"\\img", mode = 'ML_' + str(Kfold_cnt))
            ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, confusion_img = MyEstimator.confusion(feature_val_label, predict, ML_val_keyLabel, classes = CLASSNANE, logPath = logPath+"\\img", mode ='ML_' + str(Kfold_cnt))
            
            if ML_roc_auc != -1:
                ML_roc_auc = max(ML_roc_auc.values())
            ML_ACC_list.append(round(ML_Accuracy, 2))
            ML_AUC_list.append(round(ML_roc_auc, 2))
            
            logger.info("Kfold = [{}]\t".format(Kfold_cnt))
            logger.info("ACC : {:.2f} => {:.2f}\t  AUC : {:.2f} => {:.2f}".format(Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
            logger.info("SPE : {:.2f} => {:.2f}\t  SEN : {:.2f} => {:.2f}".format(Specificity, ML_Specificity, Sensitivity, ML_Sensitivity))
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
        logger.info("ACC : {:.3f} => {:.3f}\t  AUC : {:.3f} => {:.3f}".format(Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
        logger.info("SPE : {:.3f} => {:.3f}\t  SEN : {:.3f} => {:.3f}".format(Specificity.item(), ML_Specificity.item(), Sensitivity.item(), ML_Sensitivity.item()))
        logger.info("===================================================================================================")
        logger.info("ML_ACC_list :")
        logger.info(ML_ACC_list)
        logger.info("ML_AUC_list :")
        logger.info(ML_AUC_list)
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
