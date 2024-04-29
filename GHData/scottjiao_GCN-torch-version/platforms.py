




import exps
import os
path=os.path.abspath(os.path.dirname(__file__))
os.chdir(path)

exps.standard_split_GCN_exp(exptime=1,dataset='cora',description='No special description')


