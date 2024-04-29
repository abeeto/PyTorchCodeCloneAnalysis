# -*- coding: utf-8 -*-
# @Time   : 19-5-15 下午3:50
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================
import pandas as pd
import numpy as np
import os
import threading as td
from queue import Queue
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.utils.data as Data
import xlsxwriter
import time

test_addr = "/home/yyd/dataset/hacking/batch_scalar/ignore_ID_-1_1"
res_url = '/home/yyd/dataset/hacking/separateToAttackAndNormal/ignore_ID_-1_1'
datnam = 'data'
# source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/"


COLSIZ = 10
tformat = lambda s: str(s).title().ljust(COLSIZ)
# print('\n%s' % ''.join(map(tformat, vars())))

def count_data_flag(path,mark=None):
    """
    func : read data(attack status records) to train or to validate
    :param path:
    :param mark:
    :return: row(number), flag(label type of list), data(array)
    """

    data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python', encoding='utf-8')#,nrows=64*64*100 ,nrows=64*64*1
    data = data1.values.astype(np.float64)
    # data = np.reshape(data, (-1, 64, 22))
    file = os.path.basename(path)
    # print('{} has data :ndim:{} dtype:{} shape:{}'.format(file,data.ndim, data.dtype, data.shape))
    print('{} has data shaped:{}'.format(file, data.shape))
    rows = data.shape[0]
    start = 0
    row = int(rows // 64)
    end = row*64
    if mark:
        if mark == 'test':
            start = int(((rows*0.8)//64)*64)
            row = int((rows-start)//64)
            end = int(start+((rows-start)//64)*64)
        elif mark == 'train':
            row = int((rows*0.8)//64)
            end = int(row * 64)
    else:
        pass

    source_flags = data[start:end,-1].tolist()

    flags = []

    """ solve normal marked as 1"""
    if 'new_data' in path:#new_data:  attack marked as 1,存在任意個1,即標記爲1
        for r in range(row):
            num = 0.
            for item in source_flags[r*64:r*64+64]:
                if item == 1.:
                    num = 1.
                    break
            flags.append(num)

    else:# data :  normal marked as 1,沒有attack即一個0都沒有的情況,就標記爲1,否則,標記爲0
        for r in range(row):
            num = 1.
            for item in source_flags[r*64:r*64+64]:
                if item == 0.:
                    num = 0.
                    break
            flags.append(num)
    one_num = flags.count(1.)
    zero_num = flags.count(0.)
    num = len(flags)
    print('%s,one numbers:%d,zero numbers:%d,total numbers:%d'%(file,one_num,zero_num,num))
    return file,one_num,zero_num,num,flags
    # try:
    #     data = data[start:end,:-1].reshape((-1,64,21))
    # except:
    #     print('Error!!! Error!!! file name: {},data shape: {},flags size:{}'.format(file,data.shape,len(flags)))
    # print('{} start at:{} acquires labels shape:{} data shape{} done read files!!!\n'.format(file, start,len(flags),data.shape))
    # return row, flags,data


def getTrainDiscriminor(path=test_addr,mark=None):
    """
    func: write test parameter to one excel file for a type of GAN
    :param path:
    :param mark:
    :return: flag(label type of list), data(array)
    """
    files = os.listdir(path)
    lens = len(files)
    # print('operate: {}, data folder have  files'.format(mark,lens))
    pool = mp.Pool(processes=lens)

    res_path = os.path.join(res_url,datnam+'_analysis_data_construction.xlsx')
    file_urls = []
    results = []
    for i in os.listdir(path):
        if '.txt' in i:
            file_urls.append(os.path.join(path,i))
            # results.append(pool.apply(testdata,(os.path.join(path,i),mark,)))#_async
            results.append(pool.apply_async(count_data_flag,(os.path.join(path,i),mark,)))#_async
    pool.close()
    pool.join()
    # print(len(results),type(results))
    flags = []
    # data = []
    # sheet 1
    workbook = xlsxwriter.Workbook(res_path)  # 建立文件
    worksheet = workbook.add_worksheet('analysis')  # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    # rownum = 1
    headers = []
    worksheet.write_row('A1', ['filename','one_num','zero_num','total_num'])

    for i,result in enumerate(results,start=1):
        # file,one,zero,num,all_flags
        # result = result#.get()
        result = result.get()[:-1]
        headers.append(result[0])
        worksheet.write_row(row=i,col=0,data=result)
        # rownum += 1

    # sheet 2
    worksheet = workbook.add_worksheet('data')
    worksheet.write_row('A1', headers)

    for i,result in enumerate(results):
        result = result.get()[-1]
        worksheet.write_column(row=1,col=i,data=result)
    workbook.close()


def separateAttAndNor(url=None,store_url=None,fname=None):
    """
    func: separate data to Attack And Normal
    :param url: dataset file dir
    :return:None
    """
    # if 'pkl' in url:
    data1 = pd.read_pickle(url,compression='zip')

    data = data1.values.astype(np.float64)

    # attack data name
    # fname = os.path.splitext(os.path.basename(url))[0].split('_')[0]

    print('{} has data shaped:{}'.format(fname, data.shape))
    rows = data.shape[0]
    start = 0
    row = int(rows // 64)
    end = row*64

    source_flags = data[start:end,-1].tolist()

    # a_flags = []
    # n_flags = []

    # atta = np.empty((64,22))
    # nor = np.empty((64,22))

    a_count = 0
    n_count = 0

    atta_url = os.path.join(store_url,'pure_attack.csv')
    norl_url = os.path.join(store_url,'pure_normal.csv')

    # batch label,if any label 1 in a batch size of data,batch label marked as 1,separate to two part,attack and only Normal
    for r in range(row):
        num = 0
        if r % 1000 == 0:
            print('{}'.format(' '.join(map(tformat,(fname,a_count,n_count)))))
            # print(r)
        if 1. in source_flags[r*64:r*64+64] or 1 in source_flags[r*64:r*64+64]:
            num = 1
        if num:
            # attack data
            atta = pd.DataFrame(data[r*64:r*64+64,:])
            atta.to_csv(atta_url,sep=',',header=False,index=False,columns=None,mode='a',index_label=None,encoding='utf-8')
            # if a_count:
            #     atta = np.concatenate((atta,data[r*64:r*64+64,:-1]),axis=0)
            # else:
            #     atta = data[r*64:r*64+64,:-1]
            a_count += 1
            # a_flags.append(num)

        else:
            # Normal data
            nor = pd.DataFrame(data[r*64:r*64+64,:])
            nor.to_csv(norl_url,sep=',',header=False,index=False,columns=None,mode='a',index_label=None,encoding='utf-8')

            # if n_count:
            #     nor = np.concatenate((nor,data[r*64:r*64+64,:-1]),axis=0)
            # else:
            #     nor = data[r*64:r*64+64,:-1]
            n_count += 1
            # n_flags.append(num)

    # print('row:',row)

    # try:
    #     atta = atta.reshape((-1,64,21))
    #     nor = nor.reshape((-1,64,21))
    # except:
    #     print('Error!!! Error!!! can not reshape result data')

    print('{},from {} to {} acquires {} blocks,labels lengh attack|normal :{}|{},done!!!\n'.
          format(fname, start,end,row,a_count,n_count))
    return row, a_count,n_count


if __name__ == '__main__':
    print('start at:{}'.format(time.asctime(time.localtime(time.time()))))

    for i in os.listdir(test_addr):
        if 'gear' in i:
            continue
        if '.pkl' in i:
            pass
        else:
            continue
        fname = i.split('_')[0]
        store_url = os.path.join(res_url, fname)
        if not os.path.exists(store_url):
            os.makedirs(store_url)
        row, a_count,n_count = separateAttAndNor(os.path.join(test_addr,i),store_url,fname)
        for ii in os.listdir(store_url):
            ii = os.path.join(store_url,ii)
            j = os.path.splitext(ii)[0]+'.pkl'
            pd.read_csv(ii,sep=None,delimiter=',',dtype=np.float64,header=None,engine='python',encoding='utf-8').\
                to_pickle(j,compression='zip')
        print('\n')
        # break
    print('end at:{}'.format(time.asctime(time.localtime(time.time()))))

    # fenxi tongji
    # getTrainDiscriminor(mark='train')

    # results = np.zeros((4,4))
    # res_path = os.path.join(res_url,datnam+'analysis_data_construction.xlsx')
    # workbook = xlsxwriter.Workbook(res_path)  # 建立文件
    # worksheet = workbook.add_worksheet('analysis')  # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    # # rownum = 1
    # headers = []
    # worksheet.write_row('A1', ['filename','one_num','zero_num','total_num'])
    #
    # for i,result in enumerate(results,start=1):
    #     # file,one,zero,num,all_flags
    #     # result = result#.get()
    #     headers.append(result[0])
    #     worksheet.write_row(row=i,col=0,data=result)
    #     # rownum += 1
    #
    # # sheet 2
    # worksheet = workbook.add_worksheet('data')
    # results = np.ones((4,10))
    # worksheet.write_row(row=0,col=0,data=headers)
    # for i,result in enumerate(results):
    #     # result = result.get()[-1]
    #     worksheet.write_column(row=1,col=i,data=result)
    # workbook.close()