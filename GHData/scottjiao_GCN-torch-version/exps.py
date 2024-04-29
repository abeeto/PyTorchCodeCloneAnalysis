

import GCNs
import exps
import results
import log
from data_process import *



def standard_split_GCN_exp(exptime=1,dataset='cora',description='No special description'):
    model_input={}
    model_input['settings']={'dataset':dataset,'training_epoch':200,'early_stopping':10,
                           'hidden1':16,'learning_rate':0.01 ,'weight_decay':5e-1 ,
                           'dropout':0.5,
                           'all_test':False}
    model_input.update(get_dataset(model_input['settings']['dataset']))
    #GCNs.sb
    #print(dir(GCNs))
    exp1_model=GCNs.GCN_kipf(model_input)
    description='A little test for new framework on GCN'
    recorder=statistic_recorder()
    
    for i in range(exptime):
        result=exp1_model.run()
        recorder.insert(result)
    
    
    write_result(description,exp1_model.settings,recorder.statistic)













