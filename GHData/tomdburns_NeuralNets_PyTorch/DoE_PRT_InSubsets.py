"""
Looks at the subsets and gives # of DoE-PRT points
"""


import pandas as pd



def check_doe(infile):
    """checks the infile for ratio of DoE-PRT"""
    data = pd.read_csv(infile)
    sub  = data.loc[data['Purity'] >= 95.]
    sub  = sub.loc[sub['Recovery'] >= 90.]
    print(len(sub), '/', len(data))


def main():
    """main"""
    print('Purity:')
    check_doe('Subsets\\Purity_Train_subset.csv')
    print('\nRecovery:')
    check_doe('Subsets\\Recovery_Train_subset.csv')
    print('\nOverall:')
    check_doe('combined_ML_Names.csv')


if __name__ in '__main__':
    main()
