import sys
import os
# sys.path.append('../')
import numpy as np
import torch
import time
# import scipy.io as scio
import matplotlib.pyplot as plt
from torch.autograd import Variable
from readDataToGAN import *
import re
from ACGAN import discriminator as ad

GPU_ENBLE = False
DATA_TYPE = 'attack_free'

module_path = r'F:\datset_label_cmp'
test_addr = r'G:\Yangyuanda\dataset\intrusion\batch_scalar\ignore_ID_-1_1'
result_path = r'F:\datset_label_cmp\results_of_dataset_cmp'
columns_ = ['pre','N_pre','F1','acc','recall','fnr','TPR','FPR']


def writelog(content,url=None):
    # a = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/full/2019-04-17/test_logs/'
    if url == None:
        print(content)
    else:
        collect_url = './'
        if not os.path.exists(collect_url):
            os.makedirs(collect_url)
        url = os.path.join(collect_url,'{}_TestLogs.txt'.format(url))
        with open(url, 'a', encoding='utf-8') as f:
            f.writelines(content + '\n')
            print(content)


def test_attfree(path, logmark, file, test):#flags,
    """
    func: test data runs at every pkl(module)
    :param path: pkl(module) path
    :param logmark: pkl mark,determinate which module
    :param num: test data rows
    :param flags: flag of every test data
    :param test: test data
    :return:no
    #Label 0 means normal,size 1*BATCH
    # Label 1 means anormal,size 1*BATCH
    """
    print('module:',path)
    t1 = time.time()
    print('dataset:',file)
    """处理"""
    # modulename = ''
    # result = np.empty((2, 1))
    # if len(jf):
    global Dnet
    Dnet = torch.load(path,map_location='cpu')
    if GPU_ENBLE:
        pass
    else:
        Dnet = Dnet.cpu()

    TP = 0  # 1 -> 1 true positive
    TN = 0  # 0 -> 0 true negative
    FN = 0  # 1 -> 0 false negative
    FP = 0  # 0 -> 1 false positive
    import math
    def f1(l,r):
        if l == 0:
            if math.fabs(l - r) < 0.5:
                # TN
                return 0
            else:
                # FP
                return 3
        else:
            if math.fabs(l - r) < 0.5:
                # TP
                return 1
            else:
                # FN
                return 2
    total = 0
    # detail_url = './detail/{}'.format(file)
    detail_url = './detail'
    if not os.path.exists(detail_url):
        os.makedirs(detail_url)
    url_numpy = os.path.join(detail_url,'{}_test_at_{}.csv'.format(logmark,file))

    for iter, (x_, label_) in enumerate(test):
        if iter == test.dataset.__len__() // 64:
            total = iter
            break
        if GPU_ENBLE:
            x_ = x_.cuda()
            try:
                Results = Dnet(x_)
                result = Results.data.cpu().numpy()
            except:
                try:
                    Results, _ = Dnet(x_)
                    result = Results.data.cpu().numpy()
                except:
                    print('path:', path,'file:',file)
        else:
            try:
                Results = Dnet(x_)
                result = Results.data.numpy()
                # print(len(result))
            except:
                # try:
                Results, _ = Dnet(x_)
                result = Results.data.numpy()
                # except:
                #     print('path:', path,'file:',file)
        result = np.squeeze(result).tolist()
        label = np.squeeze(label_.data.numpy()).tolist()
        ll = list(map(f1, label, result))
        TN += ll.count(0)
        TP += ll.count(1)
        FN += ll.count(2)
        FP += ll.count(3)
        dat1 = pd.DataFrame({'res':result,'label':label})
        if iter:
            dat1.to_csv(url_numpy,sep=',',float_format='%.2f',header=None,index=None,mode='a',encoding='utf-8')
        else:
            dat1.to_csv(url_numpy,sep=',',float_format='%.2f',header=True,index=None,mode='a',encoding='utf-8')
    res = {}
    # 1 precision of position
    try:
        # res['pre']='{}'.format(TP/(FP+TP))
        res['pre']=TP/(FP+TP)
    except ZeroDivisionError:
        res['pre'] = 'NA'
    # 2  precision of negative
    try:
        res['N_pre']=TN/(TN+FN)
        # res['N_pre']='{}'.format(TN/(TN+FN))
    except ZeroDivisionError:
        # writelog('have no P(normaly event)',file)
        res['N_pre'] = 'NA'
    # # 3 false positive rate,index of ROC , 误报 (Type I error).
    try:
        # res['FPR']='{}'.format(FP/(FP+TN))
        res['FPR']=FP/(FP+TN)
    except ZeroDivisionError:
        res['FPR'] ='NA'
    # 4 true positive rate,index of ROC
    try:
        # res['TPR'] ='{}'.format(TP/(TP+FN))
        res['TPR'] =TP/(TP+FN)
    except ZeroDivisionError:
        # writelog('have no P(normaly event)',file)
        res['TPR'] ='NA'
    # 5 accurate
    try:
        # res['acc'] = (TP+NN)/len(flags)
        res['acc'] = (TP+TN)/(total*64)
        # results['accurate'] = accurate
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['acc'] ='NA'
    #  recall same as TPR
    try:
        res['recall'] = TP/(TP+FN)
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['recall'] = 'NA'
    # F1
    # if res['P'] == 'NA' or res['recall'] == 'NA':
    #     res['F1'] = 'NA'
    # else:
    #     try:
    #         res['F1'] ='{}'.format(2/(1/np.float64(res['P']) +1/np.float64(res['recall'])))
    #     except RuntimeWarning or RuntimeError or ZeroDivisionError or ValueError or TypeError:
    #         res['F1'] = 'NA'
    #         print('P:{},R:{}'.format(res['P'],res['recall']))
    # F1
    try:
        res['F1'] = 2*TP/(2*TP+FP+FN)
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['F1'] = 'NA'
    # false negative rate (Type II error).
    try:
        # res['fnr']= '{}'.format(FN/(FN+TP))
        res['fnr']= FN/(FN+TP)
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['fnr'] = 'NA'

    t2 = time.time()
    text = ''
    for key, item in res.items():
        text += key + ':' + str(item) + ','
    writelog('len result|label:{}|{}'.format(total*64,total*64),file)
    writelog(text,file)
    writelog('test case: {} had finshed module:{}'.format(file,logmark),file)
    writelog('time test spent :{}'.format(t2 - t1), file)
    writelog('*'*40,file)
    return res

def getModulesList(modules_path):
    """
    func: sort different modules saved at different epoch,sorted name list
    :param modules_path:
    :param mark:
    :return: different module file url(address) saved at different epoch,name list
    """
    modules = os.listdir(modules_path)
    pattern = re.compile(r'\d+\.?\d*')
    num_seq = []
    new_modul = []
    for module in modules:
        if '_D.pkl' in module:
            new_modul.append(module)
    modules = new_modul
    # print(modules)
    # print('len of modules: {}'.format(len(modules)))
    for i,module in enumerate(modules):
        jf = pattern.findall(module)
        if len(jf):
            num_seq.append(jf[0])
        else:
            # 这里必须要取值大于epoch
            num_seq.append('100000')
            mark_D = i

    num_seq = list(map(int,num_seq))
    sort_seq = sorted(num_seq)
    modules_url = []

    for s in sort_seq:
        modules_url.append(os.path.join(modules_path,modules[num_seq.index(s)]))
    sort_seq = list(map(lambda x: str(x),sort_seq))
    # print('len of modules_url: {}'.format(len(modules_url)))
    # print('len of seqs: {}'.format(len(sort_seq)))

    return modules_url, sort_seq


if __name__ == '__main__':
    """test Disciminor"""

    file_names = [os.path.splitext(file)[0] for file in os.listdir(test_addr) if 'pkl' in file]# test data name
    # print(file_names)
    test_data = DataloadtoGAN(test_addr,mark='test',label=True)

    for name in os.listdir(module_path):
        if '.py' in name or 'venv' == name or '.txt' in name:
            continue
        else:
            if 'GAN' in name :
                pass
            else:
                continue

        module_url = os.path.join(module_path, name)
        result_url = os.path.join(result_path,name)
        # print('module_url:',module_url)
        # print('result_url:',result_url)
        print('------------------------run at {}---------------------------------------------------------------------'
              .format(module_url))
        if not os.path.exists(result_url):
            os.makedirs(result_url)

        os.chdir(result_url)
        # print(module_url)
        module_urls,seqs = getModulesList(module_url)
        ress = {}
        for i in columns_:
            ress[i] = []
        for i, url in list(zip(seqs,module_urls)):
            tes = test_attfree(url,'_'.join([name,i]),DATA_TYPE,test_data)#return dict test_attfree(path, logmark, file, test)
            for key,item in tes.items():
                if key == 'NA':
                    key = None
                ress[key].append(item)
        ress['Module No.'] = seqs
        summary_url = os.path.join(result_url,name+'_analysis_summary.csv')
        data = pd.DataFrame(ress,columns=list(ress.keys()))
        data.to_csv(summary_url,sep=',',float_format='%.6f',header=True,index=True,mode='w',encoding='utf-8')
