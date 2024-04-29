"""
Compares the features and checks for self-correlations
"""


import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm
import itertools as it
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


CSV  = '..\\April 23rd 2021\\PSA Outputs\\combined_ML_Names.csv'
OMIT = ['mof', 'Etot', 'EComp'] #['mof', 'Purity', 'Recovery', 'EPSA', 'Etot', 'EComp']
TARG = 'Purity'


def reformat_cols(cols):
    """reformats the column tags"""
    new = []
    for col in cols:
        n = ''
        if 'b0' in col or 'd0' in col:
            if 'b0' in col:
                n = 'b$_0$'
            elif 'd0' in col:
                n = 'd$_0$'
            else:
                print('Error Code 0 - Invalid column name:', col)
                exit()
            if '_c' in col:
                n += ', CO$_2$'
            else:
                n += ', N$_2$'
            n += ' [bar$^{-1}$]'
            new.append(n)
        elif 'q1' in col or 'q2' in col:
            if 'q1' in col:
                n += 'Q$_1$'
            elif 'q2' in col:
                n += 'Q$_2$'
            else:
                print('Error Code 1.0 - Invalid column name:', col)
                exit()
            if '_c' in col:
                n += ', CO$_2$ [mmol / g]'
            elif '_n' in col:
                n += ', N$_2$ [mmol / g]'
            else:
                print('Error Code 1.1 - Invalid column name:', col)
                exit()
            new.append(n)
        elif 'U1' in col or 'U2' in col:
            if 'U1' in col:
                n += 'U$_1$'
            elif 'U2' in col:
                n += 'U$_2$'
            else:
                print('Error Code 2.0 - Invalid column name:', col)
                exit()
            if '_c' in col:
                n += ', CO$_2$ [kJ / mol]'
            elif '_n' in col:
                n += ', N$_2$ [kJ / mol]'
            else:
                print('Error Code 2.1 - Invalid column name:', col)
                exit()
            new.append(n)
        elif 'StructuredDensity' in col:
            new.append('Structured Density [kg / m$^3$]')
        elif 'InletTemp' in col:
            new.append('Inlet Temperature [K]')
        elif 't' == col[0]:
            new.append('t$_{%s}$ [seconds]' % col.split('t')[1])
        elif 'P' == col[0] and 'Purity' not in col and 'Prod' not in col:
            new.append('P$_{%s}$ [bar]' % col.split('P')[1])
        elif 'v0' in col:
            new.append('v$_0$ [m/s]')
        elif 'Purity' in col or 'Recovery' in col:
            new.append('%s' % col + ' [%]')
        elif 'Productivity' in col:
            new.append('Productivity [moles CO$_2$ / m$^3$ / second]')
        elif 'EPSA' in col:
            new.append('E$_{PSA}$ [kWh / tonne CO$_2$]')
        else:
            print('Error Code 3 - Invalid column name:', col)
            exit()
    return new


def main():
    """main"""
    raw  = pd.read_csv(CSV)
    cols = []
    for col in [c for c in raw]:
        if col in OMIT:
            continue
        cols.append(col)
    ncols = reformat_cols(cols)
    #cols.append(TARG)
    combos = list(it.combinations(cols, 2))
    results, all_vals, all_pairs = {}, [], []
    for i, combo in tqdm(enumerate(combos)):
        one, two = combo
        tag = '%s-%s' % (one, two)
        #print(one, two)
        r, p = pearsonr(np.array(raw[one]), np.array(raw[two]))
        results[tag] = r**2
    grid = []
    for i, icol in tqdm(enumerate(cols)):
        sub = []
        for j, jcol in enumerate(cols):
            if i == j:
                sub.append(1)
            else:
                if '%s-%s' % (icol, jcol) in results:
                    sub.append(results['%s-%s' % (icol, jcol)])
                    all_vals.append(results['%s-%s' % (icol, jcol)])
                else:
                    sub.append(results['%s-%s' % (jcol, icol)])
                    all_vals.append(results['%s-%s' % (jcol, icol)])
                all_pairs.append('%s-%s' % (ncols[i], ncols[j]))
        grid.append(sub)
    grid = np.array(grid)
    print(grid.shape, max(all_vals))
    mids = np.argsort(all_vals)[::-1]
    bpi, bpj = all_pairs[mids[0]].split('-')[0], all_pairs[mids[0]].split('-')[1]
    print(all_vals[mids[0]], bpi, bpj)
    #exit()
    g = sb.heatmap(grid, cmap='jet', cbar_kws={'label': 'Pearson R$^2$'})
    g.set_xticklabels(ncols, rotation=90)
    g.set_yticklabels(ncols)
    plt.show()


if __name__ in '__main__':
    main()
