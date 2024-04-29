#The ML model using Logistic Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

true_data = pd.read_csv("csv files/True.csv")
fake_data = pd.read_csv("csv files/Fake.csv")
true_data["class"] = 1
fake_data["class"] = 0

manual_fake = fake_data.tail(10)
manual_true = true_data.tail(10)

for i in range(23470,23481,1):
    fake_data.drop([i],axis=0,inplace=True)

for i in range(23406,23417,1):
    fake_data.drop([i],axis=0,inplace=True)

manual_testing = pd.concat([manual_fake,manual_true],axis=0)
manual_testing.to_csv("manual.csv")
merged_data = pd.concat([true_data,fake_data],axis=0) 
md = merged_data.head()

#this will remove every collumn besides the texty and the 0, 1 value
unnecessary = merged_data.drop(["subject","title","date"],axis=1)
uh = unnecessary.head()

#this is mixing the data
unnecessary = unnecessary.sample(frac=1)
uh = unnecessary.head()

x = unnecessary["text"]
y = unnecessary["class"]

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25)

vector = TfidfVectorizer()
xv_train  = vector.fit_transform(x_train)
xv_test = vector.transform(x_test)
LR = LogisticRegression()
LR.fit(xv_train,y_train)

LR.score(xv_test,y_test)

pred_LR = LR.predict(xv_test)

def output_label(n):
    if n == 0:
        return False
    elif n == 1:
        return True
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_x_test = new_def_test['text']
    new_xv_test = vector.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    
    theData = output_label(pred_LR)
    return theData


