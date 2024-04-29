#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


# In[83]:


df = pd.read_excel('input_LOG.xlsx')


# In[84]:


features = ['DEPTH', 'SP', 'RD', 'RXO', 'DEN', 'PE', 'CN', 'AC']
target = ['GR']


# In[85]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_sc_scaler = StandardScaler()
y_sc_scaler = StandardScaler()

X = x_sc_scaler.fit_transform(df[features])
y = y_sc_scaler.fit_transform(df[target])

X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[86]:


data_train = pd.concat([pd.DataFrame(X_t,columns = features),pd.DataFrame(y_t, columns = target)],axis = 1)
data_train = data_train.reset_index()
X_t = data_train[features]
y_t = data_train[target]


# In[87]:


#model optimization
import optuna


# In[88]:


def objective(trial):
    param = {
        'iterations':2000,
        "learning_rate":trial.suggest_float("learning_rate",1e-06,1),
        'use_best_model':True,
        'od_type' : "Iter",
        'od_wait' : 100,
#         'random_seed': 240,
#          "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 10),
        "depth": trial.suggest_int("max_depth", 2, 10),
        "l2_leaf_reg": trial.suggest_loguniform("lambda", 1e-8, 100),
        'eval_metric':trial.suggest_categorical("loss_function",['MAE','RMSE','R2']),
         'one_hot_max_size':1024
        }
   
    kf = KFold(n_splits = 5)
    model=CatBoostRegressor(**param)
    for train_index, test_index in kf.split(X_t):
        X_train, X_valid = X_t.loc[train_index], X_t.loc[test_index]
        y_train, y_valid = y_t.loc[train_index], y_t.loc[test_index]
        model.fit(X_train,y_train,use_best_model=True, eval_set = (X_test,y_test),silent = True)
        preds = model.predict(X_valid)
        error = mse(y_valid,preds)
    
    test_preds = model.predict(X_test)
    error = mse(y_test,test_preds)
    return error


# In[89]:


study = optuna.create_study()

study.optimize(objective,n_trials = 5)


# In[90]:


study.best_params


# In[91]:


parameters = {'learning_rate': 0.20618615641106958,
 'max_depth': 7,
 'l2_leaf_reg': 0.7916385974694847,
 'loss_function': 'MAE'}


# In[92]:


df_test = pd.read_excel('test_LOG.xlsx')
X_ = x_sc_scaler.fit_transform(df_test[features])
y_ = y_sc_scaler.fit_transform(df_test[target])
data_test = pd.concat([pd.DataFrame(X_,columns = features),pd.DataFrame(y_, columns = target)],axis = 1)
data_test = data_test.reset_index()
X_ = data_test[features]
y_ = data_test[target]


# In[93]:



data_train_t = pd.concat([pd.DataFrame(X,columns = features),pd.DataFrame(y, columns = target)],axis = 1)
data_train_t = data_train_t.reset_index()
X = data_train_t[features]
y = data_train_t[target]


# In[94]:


control=CatBoostRegressor(**parameters)
kf = KFold(n_splits = 10)
for train_index, test_index in kf.split(X):
    X_train, X_valid = X.loc[train_index], X.loc[test_index]
    y_train, y_valid = y.loc[train_index], y.loc[test_index]
    control.fit(X_train,y_train,use_best_model=True, eval_set = (X_test,y_test),silent = True)
    preds = control.predict(X_valid)
    error_val = mse(y_valid,preds)
    val_score = r2_score(y_valid,preds)
    print(f" Validation MSELoss: {val_score:{10}.6f}. Validation R2_score: {val_score:{6}.2f}")
    print("---------------------------------------------")                                                                                


# In[95]:


test_preds = control.predict(X_)
error = mse(y_,test_preds)
test_score = r2_score(y_,test_preds)
print(f"R2_score on pred_test data: {test_score:{5}.2f}")


# In[97]:


test_preds = y_sc_scaler.inverse_transform(test_preds)
y_ = y_sc_scaler.inverse_transform(y_)


# In[99]:


plt.figure(figsize = (10,10))
plt.plot(test_preds,df_test['DEPTH'].to_numpy(),c = 'r',label = 'predictions')
plt.plot(y_,df_test['DEPTH'].to_numpy(),c = 'b', label = 'real data')
plt.xlabel('GR')
plt.ylabel('DEPTH')
plt.gca().invert_yaxis()
plt.legend(loc='upper right')
plt.show()


# In[ ]:




