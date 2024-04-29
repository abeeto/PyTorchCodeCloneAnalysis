"""
Looks at the distirbution of features, with and without scaling
"""


import numpy as np
import pandas as pd
import seaborn as sb
import itertools as it
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


CSV  = '..\\April 23rd 2021\\PSA Outputs\\combined_ML_Names.csv'
OMIT = ['mof', 'Purity', 'Recovery', 'EPSA', 'Etot', 'EComp']
TARG = 'Purity'
#LOG  = ['b0_c', 'd0_c', 'q1_c', 'q2_c', 'b0_n', 'd0_n', 'q1_n', 'q2_n', 'Pint', 'Plow', 'tblow']
LOG   = ['Pint', 'Plow', 'tblow']
DOLOG = True

def main():
    """main"""
    raw  = pd.read_csv(CSV)
    cols = []
    for col in [c for c in raw]:
        if col in OMIT:
            continue
        cols.append(col)
    N = min(len(cols), 5)
    i = 1
    if DOLOG:
        N = len(LOG)
        for j, col in enumerate(LOG):
            msub   = raw.loc[raw[col] > 0]
            mini   = min(np.array(msub[col]))
            maxi   = max(np.array(raw[col]))
            apply, offset = False, mini / 100
            plt.subplot(N,2,i)
            if min(np.array(raw[col])) > 0:
                disto = np.log10(np.array(raw[col]))
            else:
                #print(min(np.array(raw[col])), offset)
                disto = np.log10(np.array(raw[col]) + offset)
                #print(min(disto))
                apply = True
            #print(max(disto), min(disto))
            plt.hist(disto, fc=(0,0,1,0.3), edgecolor='b')
            plt.title('log$_{10}$ (%s)' % col)
            i += 1
            plt.subplot(N,2,i)
            i += 1
            scaler = StandardScaler()
            sdat = scaler.fit_transform([[k] for k in np.array(disto)])
            plt.hist([h[0] for h in sdat], fc=(0,0,1,0.3), edgecolor='b')
            plt.title('log$_{10}$ (Scaled [%s])' % col)
            #if j % 4 == 0 and j > 0:
            #    plt.show()
            #    plt.clf()
            #    i = 1
        plt.show()
    else:
        for j, col in tqdm(enumerate(cols)):
            plt.subplot(N,2,i)
            plt.hist(np.array(raw[col]), fc=(0,0,1,0.3), edgecolor='b')
            plt.title(col)
            i += 1
            plt.subplot(N,2,i)
            i += 1
            scaler = StandardScaler()
            sdat = scaler.fit_transform([[i] for i in np.array(raw[col])])
            plt.hist([i[0] for i in sdat], fc=(0,0,1,0.3), edgecolor='b')
            plt.title('Scaled [%s]' % col)
            if j % 4 == 0 and j > 0:
                plt.show()
                plt.clf()
                i = 1
        plt.show()


if __name__ in '__main__':
    main()
