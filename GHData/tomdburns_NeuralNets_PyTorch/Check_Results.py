"""
Displays results from cyclops
"""


import numpy as np
import pickle as pkl
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


#FILE = 'Prod DoE Model\\NewResults_v3.pkl'
#FILE = 'Purity Model\\NewResults_v3.pkl'
#FILE = 'Recovery Model\\NewResults_v3.pkl'
FILE = 'EPSA DoE Model\\NewResults_v3.pkl'

N    = 20000
ADJ  = True

CONVERT = (True, 60*60*24*44.01/1000/1000)


#LABEL = 'Purity [%]'
#LABEL = 'Recovery [%]'
LABEL = 'Parasitic Energy [kWh / tonne CO$_2$]'
#LABEL = 'Productivity [TPD CO$_2$ / m$^3$]'


def pearsonr2(A, B):
    """calcualtes the pearson r2"""
    a, b = np.mean(A), np.mean(B)
    top, bot1, bot2 = [], [], []
    for i, val in enumerate(A):
        x = (val - a)*(B[i] - b)
        y = (val - a)**2
        z = (B[i] - b)**2
        top.append(x)
        bot1.append(y)
        bot2.append(z)
    top, bot1, bot2 = sum(top), sum(bot1), sum(bot2)
    r = top / ((bot1*bot2)**0.5)
    return r**2


def mean_abs_err(A, B):
    """calculates the MAD"""
    errs = []
    for i, a in enumerate(A):
        d = abs(a - B[i])
        errs.append(d)
    return np.mean(errs)


def main():
    """main"""

    try:
        x, y, r = pkl.load(open(FILE, 'rb'))
    except:
        x, y, r, pY, Y = pkl.load(open(FILE, 'rb'))
    
    print(len(Y))
    nY = []
    if ('Purity' in FILE or 'Recovery' in FILE) and ADJ:
        for val in pY:
            nY.append([min(val[0], 100.)])
        pY = np.array(nY)

    if CONVERT[0] and 'Prod' in FILE:
        Y  =  Y * CONVERT[1]
        pY = pY * CONVERT[1]
    
    mad = mean_abs_err(np.array([i for i in Y], dtype='float'), np.array([i for i in pY], dtype='float'))
    print(mad)
    exit()


    #plt.subplot(121)
    #plt.plot(x, y)
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.subplot(122)
    plt.xlabel('Actual %s' % LABEL)
    plt.ylabel('Predicted %s' % LABEL)
    plt.hexbin(Y, pY, bins='log', mincnt=1, cmap='jet')
    r = pearsonr2(np.array([i for i in Y], dtype='float'), np.array([i for i in pY], dtype='float'))
    print(r)
    plt.plot([0., 330.], [0., 330.], color='k', linestyle='--', label='R$^2$=%.2f' % r)
    #plt.legend()
    plt.xlim(min(Y), max(Y))
    plt.ylim(min(pY), max(pY))
    plt.show()


if __name__ in '__main__':
    main()
