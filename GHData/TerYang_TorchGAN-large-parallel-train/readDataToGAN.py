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
from torch.autograd import Variable
# from LSGAN import discriminator as dc


BATCH_SIZE = 64

# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/"
# test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"
test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"

source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/"
# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/instrusion-dataset/test_data/data/"

### get GAN train data  old ###########################################
def minbatch_test():

    file = 'Attack_free_dataset2_ID_Normalize.txt'
    url = os.path.join(source_addr, file)
    data1 = pd.read_csv(url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=64*64*100)
    data1 = data1.values.astype(np.float32)#
    # print(data1.shape)
    data1 = np.reshape(data1, (-1, 64, 22))
    print('normal :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    num1 = data1.shape[0]
    return num1,data1

def get_data():#train_normal,train_anormal,,num=64*10000

    files = os.listdir(source_addr)
    normals = []
    for file in files:
        normals.append( os.path.join(source_addr,file))
    # print('normals lenght:',normals)

    # normal0_name = os.path.basename(normals[0])
    # normal1_name = os.path.basename(normals[1])
    print('dataset:\n',files)
    # exit()
    data1 = pd.read_csv(normals[0], sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data1 = data1.values.astype(np.float32)#

    # data2 = pd.read_csv(train_anormal, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=num)
    data2 = pd.read_csv(normals[1], sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data2 = data2.values.astype(np.float32)#,copy=True
    print('normal1 :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    # print('finished:{}'.format(normal0_name))

    print('normal2 :ndim:{} dtype:{} shape:{}'.format(data2.ndim, data2.dtype, data2.shape))
    # print('finished:{}'.format(normal1_name))
    num1 = data1.shape[0]//64 #int(
    num2 =  data2.shape[0]//64

    data = np.concatenate((data1[:64*num1,:],data2[:64*num2,:]),axis=0)

    # data = np.reshape(data[:num*64,],(-1,64,22)).astype(np.float32)
    data = np.reshape(data, (-1, 64, 22))
    print('data :ndim:{} dtype:{} shape:{}'.format(data.ndim, data.dtype, data.shape))
    print("normal total has {}+{}={} blocks".format(num1,num2,num1+num2))
    print('done read files!!!\n')
    return num1+num2,data

def new_get_norlmal():

    files = os.listdir(source_addr)
    normals = []
    for file in files:
        normals.append(os.path.join(source_addr, file))

    print('dataset:\n', files)
    # exit()
    data1 = pd.read_csv(normals[0], sep=None, header=None, dtype=np.str, engine='python', encoding='utf-8')#,nrows=64*64*100
    data1 = data1.values.astype(np.float32)  #
    # print('normal1 :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    num1 = data1.shape[0] // 64  # int(
    data = np.reshape(data1[:num1*64,],(-1,64,21)).astype(np.float32)
    print('data :ndim:{} dtype:{} shape:{}'.format(data.ndim, data.dtype, data.shape))
    print("normal total has {} blocks".format(num1))
    print('done read files!!!\n')
    return num1 , data

### get GAN test data  new ###########################################
def testdata(path,mark=None):
    """
    func : read data(attack status records) to train or to validate
    :param path:
    :param mark:
    :return: row(number), flag(list of label of every batch size), data(array)
    """
    # path_AT = ''
    if 'pkl' in path:
        data1 = pd.read_pickle(path,compression='zip')
    else:
        data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python', encoding='utf-8')#,nrows=64*64*100 ,nrows=64*64*1

    data = data1.values.astype(np.float64)
    # data = np.reshape(data, (-1, 64, 22))

    # attack data name
    file = os.path.splitext(os.path.basename(path))[0].split('_')[0]

    # print('{} has data :ndim:{} dtype:{} shape:{}'.format(file,data.ndim, data.dtype, data.shape))
    print('{} has data shaped:{}'.format(file, data.shape),end=',')
    rows = data.shape[0]
    start = 0
    row = int(rows // 64)
    end = row*64
    if mark:
        '''完全训练的数据集'''
        if mark == 'test':
            start = int(((rows * 0.99) // 64) * 64)
            row = int((rows * 0.01) // 64)
            # end = int(start + ((rows - start) // 64) * 64)
            end = int(start + row * 64)
        elif mark == 'train':
            print('get type:%s' % 'train')
            row = int((rows * 0.98) // 64)
            end = int(row * 64)
        elif mark == 'validate':
            print('get type:%s' % 'validate')
            row = int((rows * 0.01) // 64)
            start = int(((rows * 0.98) // 64) * 64)
            end = int(start + row * 64)
            print('row:{},row%64={}|{}'.format(row, row % 64, (end - start) % 64))
    else:
        print('It is illegal that mark is None!!!')

    source_flags = data[start:end,-1].tolist()

    flags = []

    # batch label,if any label 1 in a batch size of data,batch label marked as 1
    for r in range(row):
        num = 0
        if 1. in source_flags[r*64:r*64+64] or 1 in source_flags[r*64:r*64+64]:
            num = 1
            # flags.append(num)
            # continue
        flags.append(num)
    # print('row:',row)
    try:
        data = data[start:end,:-1].reshape((-1,64,21))

    except:
        print('Error!!! Error!!! file name: {},data shape: {},flags size:{}'.format(file,data.shape,len(flags)))
    print('{} start at:{},end at:{} acquires labels lengh:{} data size:{} done read files!!!\n'.format(file, start,end,len(flags),data.shape))
    return row, flags,data,file

### get first discriminor data ###########################################
def getTrainDiscriminor(path=test_addr,mark=None):
    """
    func: read attack data to train module such as Discriminator(independently) through multiprocessing
    :param path:
    :param mark:
    :return: flag(label type of list), data(array)
    """
    print('-----------------------------------%s,%s-----------------------------'%(getTrainDiscriminor.__name__,mark))
    print('data address %s'%path)
    files = [os.path.join(path,f) for f in os.listdir(path) if 'pkl' in f]
    lens = len(files)
    # print('operate: {}, data folder have  files'.format(mark,lens))

    results = []
    if lens==0:
        files= [os.path.join(path, i) for i in os.listdir(path) if '.txt' in i]
    pool = mp.Pool(processes=len(files))

    for i in files:
        # print(i)
        # results.append(pool.apply(testdata,(os.path.join(path,i),mark,)))#_async
        results.append(pool.apply_async(testdata, (i, mark,)))  # _async
    pool.close()
    pool.join()
    # print(len(results),type(results))

    flags = []
    data = []
    names = []
    row = 0
    if mark == 'test':
        for i, result in enumerate(results):
            # result = result#.get()
            result = result.get()
            # print(type(result),len(result))
            # print(result[1],result[2])

            flags.append(result[1])
            data.append(result[2])
            names.append(result[3])
            row += result[0]
        print('return data shape:{},labels shape:{},block size:{}'.format(data.__len__(), len(flags), row))

    else:
        for i,result in enumerate(results):
            # result = result#.get()
            result = result.get()
            # print(type(result),len(result))
            # print(result[1],result[2])
            if i:
                flags.extend(result[1])
                data =np.concatenate((data,result[2]))
            else:
                flags = result[1]
                data = result[2]
            names.append(result[3])
            row += result[0]
        print('return data shape:{},labels shape:{},block size:{}'.format(data.shape, len(flags), row))

    # if mark == 'validate':
    #     path_AT = '/home/gjj/dataset/intrusion/batch_scalar/ignore_ID_(-1,1)/Attack_free_dataset2.pkl'
    #     data2 = pd.read_pickle(path_AT, compression='zip')
    #     data2 = data2.loc[data2.shape[0]-row*64:,:].values.reshape((-1,64,21))
    #     data = np.concatenate((data,data2),axis=0)
    #
    #     print('1 count:%d,0 count:%d'%(flags.count(1),flags.count(0)))
    #     print('add before:',flags.__len__())
    #     flags = np.concatenate((np.array(flags).reshape((-1,1)), np.ones((row,1)))).astype(np.int)
    #     print('add affter:',flags.shape)
    #     flags = np.squeeze(flags).tolist()
    #     zero = 0
    #     one = 0
    #     for i in flags:
    #         if i == 0. or i== 0:
    #             zero += 1
    #         elif i==1. or i ==1:
    #             one += 1
    #     print('1:%d,0:%d'%(one,zero))
    #     print('flag2:',len(flags))
        # flags3 = flags.extend(flags2)

    print('------------------------------------------------------------------')
    return flags,data,names


"""get data to new GAN"""
def testToGAN(path,mark=None):
    """
    func: read normal data to train GAN defined as the Class(new gan code)
    :param path:
    :param mark:
    :return: dataloader
    """
    # 旧数据
    # files = os.listdir(path)
    # if len(files)>1:
    #     print('dataset address error at testToGAN')
    #     return -1
    # else:
    #     path = os.path.join(path,files[0])
    # data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python', encoding='utf-8')#,nrows=64*64*100
    # data = data1.values.astype(np.float64)
    # file = os.path.basename(path)
    # print('{} has data shaped:{}'.format(file, data.shape))
    # rows = data.shape[0]
    # start = 0
    # end = rows
    # row = int(rows // 64)
    # if mark:
    #     if mark == 'test':
    #         start = int(((rows*0.8)//64)*64)
    #         row = int((rows-start)//64)
    #         end = int(start+((rows-start)//64)*64)
    #     elif mark == 'train':
    #         row = int((rows*0.8)//64)
    #         end = int(row * 64)
    # else:
    #     print('arise error at testToGAN parameter mark')
    #
    # data = data[start:end,].reshape((-1,64,21))
    #
    #
    # TraindataM = torch.from_numpy(data).float()    # transform to float torchTensor
    #
    # TraindataM = torch.unsqueeze(TraindataM,1)
    # print(TraindataM.shape)
    # TorchDataset = Data.TensorDataset(TraindataM)
    #
    # # Data Loader for easy mini-batch return in training
    # train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
    # print('{} start at:{} acquires data shape{} done read files!!!\n'.format(file, start,data.shape))
    # return train_loader#, Variable(TestdataM),Variable(TestLabelM)

    # 新数据
    files = [os.path.join(path,f) for f in os.listdir(path) if 'pkl' in f]
    fl = []
    # data2 = np.empty((64,21))

    for i,f in enumerate(files):
        data1 = pd.read_pickle(f,compression='zip')
        data = data1.values.astype(np.float64)
        # file = os.path.basename(path)
        rows = data.shape[0]
        start = 0
        end = rows
        row = int(rows // 64)
        file = os.path.splitext(os.path.basename(f))[0]
        fl.append(file)
        if mark:
            if mark == 'test':
                start = int(((rows*0.8)//64)*64)
                row = int((rows-start)//64)
                end = int(start+((rows-start)//64)*64)
            elif mark == 'train':
                row = int((rows*0.8)//64)
                end = int(row * 64)
        else:
            print('arise error at testToGAN parameter mark')

        data = data[start:end, ].reshape((-1, 64, 21))
        print('{} shaped:{},trunked:{}'.format(file, data1.shape,data.shape))
        print('{} start at:{} acquires row:{},end:{} done read files!!!\n'.format(file, start, row,end))

        if i:
            data2 = np.concatenate((data2,data),axis=0)
        else:
            data2 = data

    TraindataM = torch.from_numpy(data2).float()    # transform to float torchTensor

    TraindataM = torch.unsqueeze(TraindataM,1)
    # print(TraindataM.shape)
    TorchDataset = Data.TensorDataset(TraindataM)

    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
    print('{},size:{} done read files!!!\n'.format(fl, TraindataM.shape))
    return train_loader#, Variable(TestdataM),Variable(TestLabelM)

"""get data to testing new GAN"""
def testNormal(path, mark=None):
    """
    func:read Normal data to test.py for validating the trained modules
    :param path:
    :param mark:
    :return: row(number), flag(label type of list), data(array)
    """
    files = os.listdir(path)
    if len(files) > 1:
        print('dataset address error at testToGAN')
        return -1
    else:
        path = os.path.join(path, files[0])
    data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python',
                        encoding='utf-8')  # ,nrows=64*64*100
    data = data1.values.astype(np.float64)
    # data = np.reshape(data, (-1, 64, 22))
    file = os.path.basename(path)
    # print('{} has data :ndim:{} dtype:{} shape:{}'.format(file,data.ndim, data.dtype, data.shape))
    print('{} has data shaped:{}'.format(file, data.shape))
    rows = data.shape[0]
    start = 0
    end = rows
    row = int(rows // 64)
    if mark:
        if mark == 'test':
            start = int(((rows * 0.8) // 64) * 64)
            row = int((rows - start) // 64)
            end = int(start + ((rows - start) // 64) * 64)
        elif mark == 'train':
            row = int((rows * 0.8) // 64)
            end = int(row * 64)
    else:
        print('arise error at testToGAN parameter mark')

    data = data[start:end, ].reshape((-1, 64, 21))
    # num = data.shape[0]
    # print('num:',)
    flag = np.zeros((row,1)).tolist()
    print('{} start at:{} acquires data shape{},flag numbers:{} done read files!!!\n'.format(file, start,data.shape,len(flag)))
    return row, flag, data


# 大数据下训练:验证:测试比例 98:1:1
def DataloadtoGAN(path,mark=None,label=False,single_dataset=False,hacking=False):
    """
    func: read normal data to train GAN defined as the Class(new gan code)
    :param path:dataset url
    :param mark:'validate' 'test' 'train'
    :param label: whether deliver label
    :param single_dataset:  for whole normal dataset,not suitable for normal status dataset from hacking dataset
    :param hacking: whole normal dataset or  normal status dataset from hacking dataset
    :return: dataloader,torch.Tensor

    """

    if mark == None:
        print('mark is None, please checks')
        return
    if hacking:
        files = []
        for d in os.listdir(path):
            if '.' in d:
                continue
            for f in os.listdir(os.path.join(path, d)):
                if 'normal.pkl' in f:
                    files.append(os.path.join(path,d, f))
    else:
        files = [os.path.join(path,f) for f in os.listdir(path) if 'pkl' in f]

    fl = []
    if single_dataset:
        files = [i for i in files if 'Attack_free_dataset2' in i]
    data2 = np.empty((64,21))
    atta2 = np.empty((64,21))

    # read dataset
    for i,f in enumerate(files):
        atta = np.empty(((64,21)))

        data1 = pd.read_pickle(f,compression='zip')
        data = data1.values.astype(np.float64)
        # file = os.path.basename(path)
        rows = data.shape[0]
        start = 0
        end = rows
        row = int(rows // 64)
        row1 = row
        file = os.path.splitext(os.path.basename(f))[0]
        fl.append(file)
        dirname = os.path.dirname(f).split('/')[-1]

        if mark == 'test':
            start = int(((rows*0.99)//64)*64)
            row = int((rows*0.01)//64)
            if start % 64 == 0:
                pass
            else:
                start = ((start // 64) + 1) * 64
            # end = int(start+((rows-start)//64)*64)
            end = int(start+row*64)
        elif mark == 'train':
            print('get type:%s'%'train')
            # row = int((rows*0.01)//64)
            # end = int(row * 64)
            row = int((rows*0.98)//64)
            end = int(row * 64)
        elif mark == 'validate':
            print('get type:%s,datatype:%s'%('validate',dirname))
            row = int((rows*0.01)//64)
            start = int(((rows*0.98)//64) * 64)
            if start % 64 == 0:
                pass
            else:
                start = int(((start // 64) + 1) * 64)
            end = int(start + row*64)
            # end =int(((rows*0.99)//64) * 64)

        if hacking:
            data = data[start:end, :-1].reshape((-1, 21))
            if mark == 'validate':
                url = os.path.dirname(f) + '/pure_attack.pkl'
                atta = pd.read_pickle(url,compression='zip')
                atta = pd.DataFrame(atta).to_numpy().reshape((-1,64,22))
                print('{},shape:{}'.format('pure_attack',atta.shape),end=',')
                atta = atta[:row,:,:21]
                # print('atta.shape---:',atta.shape)
                print('start at:{},end:{},acquires row:{},done read files!!!'.format(start, end, row))
            if i > 0:
                data2 = np.concatenate((data2, data), axis=0).reshape((-1, 21))
                atta2 = np.concatenate((atta2, atta), axis=0)
                # print('atta2.shape:',atta2.shape)
            else:
                data2 = data
                atta2 = atta
        else:
            data = data[start:end, :].reshape((-1, 21))
            if i > 0:
                data2 = np.concatenate((data2,data),axis=0).reshape((-1,21))
            else:
                data2 = data
        print('{} shaped:{},trunked:{}'.format(file, data1.shape, data.shape),end=',')
        print('get|all:{}|{},blocks:{}'.format(row, row1, row % 64),end=',')
        print('start at:{},end:{},percent:{}%,done read files!!!\n'.format(start, end, float(row/row1)))
        # exit()
    if mark == 'validate':
        atta2 = atta2.reshape((-1,64,21))
        data2 = data2.reshape((-1, 64, 21))
        label1 = np.ones((atta2.shape[0],1))
        label0 = np.zeros((data2.shape[0],1))

        data2 = np.concatenate((data2,atta2),axis=0)
        labels = np.concatenate((label0,label1),axis=0)

        TraindataM = torch.from_numpy(data2).float()  # transform to float torchTensor
        TraindataM = torch.unsqueeze(TraindataM, 1)
        Traindata_LabelM = torch.from_numpy(labels).float()
        TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)

        print('{},size:{} label:{},done read files!!!\n'.format('validate mix dataset', TraindataM.shape, label))
        return Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    TraindataM = torch.from_numpy(data2.reshape((-1, 64, 21))).float()    # transform to float torchTensor
    TraindataM = torch.unsqueeze(TraindataM,1)

    if label:
        # if mark == 'train' or mark == 'test':
        labels = np.zeros((TraindataM.shape[0],1))
        Traindata_LabelM = torch.from_numpy(labels).float()
        TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
        print('{},size:{} label:{},done read files!!!\n'.format(fl, TraindataM.shape, label))
        return Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    else:
        # if mark == 'train' or mark == 'test':
        # Data Loader for easy mini-batch return in training
        TorchDataset = Data.TensorDataset(TraindataM)
        print('{},size:{} label:{},done read files!!!\n'.format(fl, TraindataM.shape, label))
        return Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)


# addr = '/home/yyd/dataset/hacking/separateToAttackAndNormal/ignore_ID_-1_1/'#attack data
# files = []
# for d in os.listdir(addr):
#     if '.' in d:
#         continue
#     for f in os.listdir(os.path.join(addr, d)):
#         if 'normal.pkl' in f:
#             files.append(os.path.join(addr, d, f))
# for i in files:
#     print(i)

# a = DataloadtoGAN(addr, mark='train',hacking=True,label=True)
# print(a.dataset.__len__())
# print(len(a.dataset.tensors))