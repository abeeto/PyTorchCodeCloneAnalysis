"""
Plots the distributions
"""


import numpy as np
import pandas as pd
import pickle as pkl
from os.path import exists
import matplotlib.pyplot as plt


CONVERT = (True, 60*60*24*44.01/1000/1000)


def plot_pure():
    """plots the purity and recovery distritbutions for all points in set"""
    pfile    = 'All_PurityRecovery.pkl'
    if exists(pfile):
        purity, recovery = pkl.load(open(pfile, 'rb'))
    else:
        infile   = 'combined_ML_Names.csv'
        data     = pd.read_csv(infile)
        purity   = np.array(data['Purity'])
        recovery = np.array(data['Recovery'])
        pkl.dump((purity, recovery), open(pfile, 'wb'))

    plt.subplot(121)
    plt.hist(purity, fc=(0.,0.,1.,0.2), edgecolor='b')
    plt.xlabel('Purity of Captured CO$_2$ [%]')
    #plt.yticks([])
    plt.ylabel('Number of Datapoints [x10$^6$]')

    plt.subplot(122)
    plt.hist(recovery, fc=(0.,0.,1.,0.2), edgecolor='b')
    plt.xlabel('Recovery of CO$_2$ from Flue Gas [%]')
    #plt.yticks([])
    plt.ylabel('Number of Datapoints [x10$^6$]')

    plt.show()
    plt.clf()

    return purity, recovery


def plot_dists(all, tag=None, col=None, plabel=None):
    """plots the new purity distributions"""
    pfile  = '%s_Training.csv' % tag
    if exists(pfile):
        train = pkl.load(open(pfile, 'rb'))
    else:
        infile = 'Subsets\\%s_Train_subset.csv' % tag
        raw    = pd.read_csv(infile)
        train  = np.array(raw[col])
        pkl.dump(train, open(pfile, 'wb'))

    #plt.subplot(121)
    #plt.hist(all, fc=(0.,0.,1.,0.2), edgecolor='b')
    #plt.xlabel(plabel)
    #plt.yticks([])
#
    #plt.subplot(122)
    plt.hist(train, fc=(0.,0.,1.,0.2), edgecolor='b')
    plt.xlabel(plabel)
    plt.ylabel('Number of Datapoints')
    #plt.yticks([])

    print('N %s:' % tag, len(train))

    plt.show()


def get_percents(col='Purity', r1=90., r2=50.):
    """gets the presents within certain ranges"""
    pfile    = 'All.pkl'
    if exists(pfile):
        data = pkl.load(open(pfile, 'rb'))
    else:
        infile   = 'combined_ML_Names.csv'
        data     = pd.read_csv(infile)
        pkl.dump(data, open(pfile, 'wb'))
    sub1 = data.loc[data[col] > r1]
    sub2 = data.loc[data[col] < r2]

    Ntot  = len(data)
    Nsub1 = len(sub1)
    Nsub2 = len(sub2)

    perc1 = 100 * Nsub1 / Ntot
    perc2 = 100 * Nsub2 / Ntot

    print('N %s values above: %.1f' % (col, r1), '%:', '%.2f' % perc1, '%')
    print('N %s values below: %.1f' % (col, r2), '%:', '%.2f' % perc2, '%')
    

def import_doe():
    """import and plot the parasitic energy distributions"""
    infile = 'Subsets\\DoE_Only.csv'
    data   = pd.read_csv(infile)

    print('N:', len(data))

    plt.clf()
    plt.hist(data['EPSA'], fc=(0.,0.,1.,0.2), edgecolor='b') #, bins=1000)
    #plt.yticks([])
    plt.xlabel('Parasitic Energy [kWh / tonne CO$_2$]')
    plt.ylabel('Number of Datapoints')
    plt.show()
    plt.clf()
    #n = np.array(data['EPSA'])
    #k = np.argsort(n)
    #plt.plot([i for i in range(len(k))], [n[i] for i in k], color='b')
    #plt.show()

    plt.clf()
    if CONVERT[0]:
        plt.hist(np.array(data['Productivity'])*CONVERT[1], fc=(0.,0.,1.,0.2), edgecolor='b') #, bins=1000)
        unit = 'TPD CO$_2$ / m$^3$'
    else:
        plt.hist(data['Productivity'], fc=(0.,0.,1.,0.2), edgecolor='b') #, bins=1000)
        unit = 'moles CO$_2$ / m$^3$ / s'
    #plt.yticks([])
    plt.ylabel('Number of Datapoints')
    plt.xlabel('Productivity [%s]' % unit)
    plt.show()
    plt.clf()
    #n = np.array(data['Productivity'])
    #k = np.argsort(n)
    #plt.plot([i for i in range(len(k))], [n[i] for i in k], color='b')
    #plt.show()


def main():
    """main"""
    pure, recv = plot_pure()
    #get_percents()
    #plot_dists(pure, tag='Purity', col='Purity', plabel='Purity of Captured CO$_2$ [%]')
    #plot_dists(recv, tag='Recovery', col='Recovery', plabel='Recovery of CO$_2$ from Flue Gas [%]')

    #import_doe()


if __name__ in '__main__':
    main()
