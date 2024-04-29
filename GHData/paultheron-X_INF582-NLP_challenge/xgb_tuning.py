from math import gamma
import pandas as pd
import numpy as np
import csv
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score


# Read The data
training_set = pd.read_json('processed_data/train_set.json')
test_set = pd.read_json('processed_data/test_set.json')

roberta_train = pd.read_csv("processed_data/roberta_train.csv")[["label"]]
roberta_test = pd.read_csv("processed_data/roberta_test.csv")[["label"]]
roberta_train.rename(columns={"label": "roberta"},inplace=True)
roberta_test.rename(columns={"label": "roberta"},inplace=True)

gltr_train = pd.read_csv("processed_data/gltr_train.csv")
gltr_test = pd.read_csv("processed_data/gltr_test.csv")

keywords_train = pd.read_csv("processed_data/keywords_train.csv")
keywords_test = pd.read_csv("processed_data/keywords_test.csv")

embedding_train = pd.read_csv("processed_data/embedding_train.csv")
embedding_test = pd.read_csv("processed_data/embedding_test.csv")

ngrams_train = pd.read_csv("processed_data/ngrams_train.csv")
ngrams_test = pd.read_csv("processed_data/ngrams_test.csv")

rouge_train = pd.read_csv("processed_data/rouge_train.csv")
rouge_test = pd.read_csv("processed_data/rouge_test.csv")

# Combining
X = pd.concat([roberta_train, gltr_train, keywords_train, embedding_train, ngrams_train, rouge_train], axis = 1)
Y = training_set.label

X_train, X_val , Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

dtrain = xgboost.DMatrix(X_train, label = Y_train)
dtest = xgboost.DMatrix(X_val, label = Y_val)

params = {
    'objective' : 'binary:logistic', 
    'max_depth' : 6, 
    'min_child_weight' : 1,
    'eta': 0.3,
    'subsample' : 1,
    'colsample_bytree' : 0.7,
    'learning_rate' : 0.1, 
    #'n_estimators' : 100, 
    'gamma' : 1,
    #'use_label_encoder' :False, 
    'eval_metric' : 'error', 
    'gamma' : 1,
    
}

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(2,12)
    for min_child_weight in range(1,8)
]
def tune_depth():
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
                                max_depth,
                                min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results =xgboost.cv(
            params,
            dtrain,
            num_boost_round=45,
            seed=42,
            nfold=5,
            metrics={'error'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_error = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tError {} for {} rounds".format(mean_error, boost_rounds))
        if mean_error < min_mae:
            min_mae = mean_error
            best_params = (max_depth,min_child_weight)
    return best_params, min_mae

best_params, min_mae = tune_depth()
#best_params, min_mae = [5,2], 0.034027800000000004

print("Best params: {}, {}, Error: {}".format(best_params[0], best_params[1], min_mae))

params['max_depth'] = best_params[0] #5
params['min_child_weight'] = best_params[1] #2

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(3,11)]
    for colsample in [i/10. for i in range(3,11)]
]

def tune_sample():
    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(
                                subsample,
                                colsample))
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results =xgboost.cv(
            params,
            dtrain,
            num_boost_round=45,
            seed=42,
            nfold=5,
            metrics={'error'},
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,colsample)
    return best_params, min_mae

best_params, min_mae = tune_sample()
#best_params, min_mae = [0.5, 0.8], 0.033888800000000004

print("Best params: {}, {}, Error: {}".format(best_params[0], best_params[1], min_mae))

params['subsample'] = best_params[0] #5
params['colsample_bytree'] = best_params[1] #2


def tune_eta():
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results =xgboost.cv(
                params,
                dtrain,
                num_boost_round=45,
                seed=42,
                nfold=5,
                metrics=['error'],
                early_stopping_rounds=10
            )
        # Update best score
        mean_mae = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    return best_params, min_mae

best_params, min_mae = tune_eta()

print("Best params: {}, error: {}".format(best_params, min_mae))

params['eta'] = best_params

def tune_learning_rate():
    min_mae = float("Inf")
    best_params = None
    for lr in [1, .1, .01, .001]:
        print("CV with lr={}".format(lr))
        # We update our parameters
        params['learning_rate'] = lr
        # Run and time CV
        cv_results =xgboost.cv(
                params,
                dtrain,
                num_boost_round=45,
                seed=42,
                nfold=5,
                metrics=['error'],
                early_stopping_rounds=10
            )
        # Update best score
        mean_mae = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = lr
    return best_params, min_mae

best_params, min_mae = tune_learning_rate()

print("Best params: {}, Error: {}".format(best_params, min_mae))

params['learning_rate'] = best_params

with open('processed_data/xgboost_tuned_params.txt', 'w') as f:
    f.write(str(params))
    
print('Best params saved')