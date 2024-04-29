"""
Considers the models' training sets and calculates the domains for all
material specific parameters
"""


import numpy as np
import pandas as pd
from tqdm import tqdm


DIR = 'Subsets\\'
PUR = DIR + 'Purity_Train_subset.csv'
REC = DIR + 'Recovery_Train_subset.csv'
PE  = DIR + 'DoE_Only.csv'
PR  = DIR + 'DoE_Only.csv'


OMIT = ['mof', 'InletTemp', 'tads', 'tblow', 'tevac', 'Pint', 'Plow', 'v0', 'Purity', 'Recovery', 'Productivity', 'Etot', 'EPSA', 'EComp']


# ================================================
# Constants
# ================================================
ATM = 1.01325                   # bar
RG  = 8.20573660809596*10**(-5) # m3 atm / K / mol
RGB = RG / ATM                  # m3 bar / K / mol
T   = 298.15                    # Kelvin
RT  = RGB * T                   # m3 bar / mol

KCAL_TO_J = 4184
# ================================================


def conv(b):
    """converts a b from (1/bar) to (1/c)"""
    return b * RT


def conv2(b):
    """converts a b from (1/c) to (1/bar)"""
    return b / RT


def b_to_b0(b, U):
    """Converts b values to b0 values"""
    #b = b*(1/OPTIONS.get_float('PA'))
    r = 8.314473
    t = 298.15
    rt = r*t
    ev = np.exp(-U/rt)
    return b/ev


def b0_to_b(b0, U):
    """Converts b0 values to b values"""
    #b = b*(1/OPTIONS.get_float('PA'))
    r = 8.314473
    t = 298.15
    rt = r*t
    ev = np.exp(-U/rt)
    return b0*ev


def HoA_to_U(HoA):
    """Converts the HoA to U"""
    HoA = -1*HoA*KCAL_TO_J
    r = 8.314473
    t = 298.15
    rt = r*t
    return HoA + rt


def U_to_HoA(U):
    """Converts the HoA to U"""
    r = 8.314473
    t = 298.15
    rt = r*t
    H = U - rt
    return -1 * H / KCAL_TO_J


def flip(data, atom):
    """Flips the parameters"""
    b, d, q1, q2 = 'b0_%s' % atom, 'd0_%s' % atom, 'q1_%s' % atom, 'q2_%s' % atom
    b_, d_, q1_, q2_ = [], [], [], []
    for i in tqdm(data.index):
        if data[b][i] >= data[d][i]:
            b_.append(data[b][i])
            d_.append(data[d][i])
            q1_.append(data[q1][i])
            q2_.append(data[q2][i])
        else:
            b_.append(data[d][i])
            d_.append(data[b][i])
            q1_.append(data[q2][i])
            q2_.append(data[q1][i])
    data = data.drop([b, d, q1, q2], axis=1)
    data[b]  = np.array(b_)
    data[d]  = np.array(d_)
    data[q1] = np.array(q1_)
    data[q2] = np.array(q2_)
    return data


def adjust(data):
    """fixes issues in arrays"""
    print('Running Check...')
    sub_c = data.loc[data['b0_c'] < data['d0_c']]
    sub_n = data.loc[data['b0_n'] < data['d0_n']]

    if len(sub_c) > 0:
        print('Carbons need flipping')
        data = flip(data, 'c')
    if len(sub_n) > 0:
        print('Nitrogens need flipping')
        data = flip(data, 'n')

    print('\nDone.')

    return data


def run_convert(d):
    """runs the conversion because reasons"""
    new = []
    for x in tqdm(np.array(d), leave=False):
        new.append(conv2(b0_to_b(x)))
    return np.array(new)


def get_range(ifile):
    """gets the domains"""
    data = pd.read_csv(ifile)

    #data['b298_c'] = conv2(b0_to_b(data['b0_c'], data['U1_c']))
    #data['d298_c'] = conv2(b0_to_b(data['d0_c'], data['U1_c']))
    #data['b298_n'] = conv2(b0_to_b(data['b0_n'], data['U1_n']))
    #data['d298_n'] = conv2(b0_to_b(data['d0_n'], data['U1_n']))
    #print('\tDone.')
    #data = data.drop(OMIT, axis=1)
    #adjust(data)
    #exit()

    data = adjust(data)

    cols = [c for c in data]
    for col in cols:
        if col in OMIT:
            continue
        print(col, 'max =', max(data[col]), 'min =', min(data[col]))


def main():
    """main"""
    #print('Purity:')
    #get_range(PUR)

    #print('\n\nRecovery:')
    #get_range(REC)

    print('\n\nPE:')
    get_range(PE)


if __name__ in '__main__':
    main()
