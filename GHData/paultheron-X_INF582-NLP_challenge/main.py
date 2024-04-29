from xgboost import plot_importance
import pandas as pd
import numpy as np
import csv
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns


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


xgbc = XGBClassifier(objective='binary:logistic', colsample_bytree= 0.7, learning_rate= 0.1, max_depth= 3, n_estimators= 1000, use_label_encoder=False, eval_metric='error')
clf = xgbc.fit(X, Y)


def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.yticks(fontsize = 8)

# plot
ax = sns.barplot()
#plt.bar(range(len(xgbc.feature_importances_)), xgbc.feature_importances_)

plot_feature_importance(xgbc.feature_importances_,X.columns,'XG BOOST ')
plt.savefig('barplot.png')
print('xgboost feature impotance saved to barplot.png')

'''y_pred_val = xgbc.predict(X_val)
y_pred_val = y_pred_val.round(0).astype(int)

print("Accuracy :", accuracy_score(Y_val, y_pred_val))
print("Accuracy without features", accuracy_score(Y_val, np.round(X_val[["roberta"]].to_numpy(),0)))'''
# Write predictions to a file
X_test = pd.concat([roberta_test, gltr_test, keywords_test, embedding_test, ngrams_test, rouge_test], axis = 1)

predictions = xgbc.predict(X_test)

with open("output/output/submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
    print("'Submission saved to output/output/submission.csv !")