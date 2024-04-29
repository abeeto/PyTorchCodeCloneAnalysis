"""
Reorganizes the output file to evenly distribute the values used in fittings
"""


import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt


INFILE = 'combined_ML_Names.csv'
PKL    = 'raw_data.pkl'
WRITE  = False

PURITY   = False
RECOVERY = False
PE       = True
PROD     = False


def plot_distributions(data):
    """plots the distribution of target variables"""
    plt.subplot(221)
    a, b, c = plt.hist(data['Purity'], fc=(0.,0.,1.,0.2), edgecolor='b')
    purity = (a,b,c)
    print('Minimum Purity Bin  :', min(a))
    plt.xlabel('Purity')

    plt.subplot(222)
    a, b, c = plt.hist(data['Recovery'], fc=(0.,0.,1.,0.2), edgecolor='b')
    print('Minimum Recovery Bin:', min(a))
    recovery = (a,b,c)
    plt.xlabel('Recovery')

    plt.subplot(223)
    a, b, c = plt.hist(data['EPSA'], fc=(0.,0.,1.,0.2), edgecolor='b')
    print('Minimum PE Bin      :', min(a))
    parae = (a,b,c)
    plt.xlabel('Parasitic Energy')

    plt.subplot(224)
    a, b, c = plt.hist(data['Productivity'], fc=(0.,0.,1.,0.2), edgecolor='b')
    print('Minimum Prod Bin    :', min(a))
    plt.xlabel('Productivity')
    prod = (a,b,c)

    #plt.show()
    plt.clf()

    return purity, recovery, parae, prod


def reorg(data, tag, info, N=None):
    """Re-organize the array to ensure balanced data"""
    n, bins, patches = info
    IDS = []
    if N is None:
        N = min(n)
    ALL = [i for i in data.index]
    for i, v in enumerate(bins):
        if i == 0:
            continue
        sub = data.loc[data[tag] >= bins[i-1]]
        sub = data.loc[data[tag] < v]
        ids = np.array([i for i in sub.index])
        np.random.shuffle(ids)
        for k, id in enumerate(ids):
            if k >= N:
                break
            IDS.append(id)

    #OTH = []
    #for i in ALL:
    #    if i not in IDS:
    #        OTH.append(i)

    new = data.loc[IDS,:]
    #oth = data.loc[OTH,:]
    oth = data.drop(IDS)

    plt.subplot(121)
    plt.title('Full Set N=%i' % len(data))
    plt.hist(data[tag], fc=(0.,0.,1.,0.2), edgecolor='b')
    plt.xlabel(tag)

    plt.subplot(122)
    plt.title('Sub Set N=%i' % len(new))
    a, b, c = plt.hist(new[tag], fc=(0.,0.,1.,0.2), edgecolor='b')
    plt.xlabel(tag)

    print('\tData Curated. New:', min(a), 'to', max(a))
    if WRITE:
        print('\tDumping to csv...')
        new.to_csv('Subsets\\%s_Train_subset.csv' % tag, index=False)
        oth.to_csv('Subsets\\%s_Other_subset.csv' % tag, index=False)
        print('\tcsv files written.')

    plt.show()


def main():
    """main"""
    os.system('cls')
    print(80*'=')

    if not os.path.exists(PKL):
        data = pd.read_csv(INFILE)
        data = data.loc[data['Etot'] <= 400]
        pkl.dump(data, open(PKL, 'wb'))
    else:
        data = pkl.load(open(PKL, 'rb'))
    pu, re, pe, pr = plot_distributions(data)

    print(80*'=')
    if PURITY:
        print('Generating Purity Files:')
        reorg(data, 'Purity', pu, N=3746)
        print('\tDone.')
    if RECOVERY:
        print('Generating Recovery Files:')
        reorg(data, 'Recovery', re, N=200000)
        print('\tDone.')
    if PE:
        print('Generating PE Files:')
        reorg(data, 'EPSA', pe, N=227062)
        print('\tDone.')
    if PROD:
        print('Generating Prod Files:')
        reorg(data, 'Productivity', pr, N=6962)
        print('\tDone.')
    print(80*'=')
    print('Done')


if __name__ in '__main__':
    main()
