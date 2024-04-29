import pickle

import os
import pandas as pd



def readUCRdata(DBname):
    
    cwd = os.getcwd()
        
    df = pickle.load(open(cwd+'/UCI_REPO/Pickle/'+DBname,'rb'))

    
    ## Some sample contains ? character. 
    
    if DBname =='ad_data':
        df = df[(df.values != '?').all(axis=1)]      
        df = df[(df.values != '   ?').all(axis=1)]

    if DBname =='arrhythmia':
        df = df[(df.values != '?').all(axis=1)]      
        df = df[(df.values != '   ?').all(axis=1)]
        
    if DBname =='dermatology':
        df = df[(df.values != '?').all(axis=1)]      
        df = df[(df.values != '   ?').all(axis=1)]


    if DBname == 'gene_seq':
        
        df = df.drop(['Unnamed: 0'], axis=1)
        
        
    
    if DBname == 'mice_data':
        
        df = df.drop(['MouseID','Genotype',  'Treatment',  'Behavior'], axis=1)
        df = df.dropna()  
        
        
    if DBname == 'Epileptic_Seizure':
        
        df = df.drop(['Unnamed: 0'], axis=1)
        
        
    data = df.values
    n_rows,n_col = data.shape
    input_data = data[:,:n_col-1]
    input_data = input_data.astype(float)
    
    return input_data

                           