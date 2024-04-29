# importing the tools we need

# Regular EDA and plotting libraries
import numpy as np
import pandas as pd

# Models from scikit learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Model Evaluations 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score


def predict(a, b, c, d, e, f, g, i, j, k, l, m, n):
    # Importing the dataset
    df = pd.read_csv('heart.csv')
    df.head()

    # splitting the data into X and y
    X = df.drop('target', axis = 1)
    y = df['target']

    # splitting data into train and test set
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Feature Scaling
    sc= StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)


    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # creating a hyperparameter grid for logistic regression
    log_reg_grid = {'C': np.logspace(-4, 4, 20),
                    'solver': ['liblinear']}

    # Tuning logisticregression

    np.random.seed(42)
    # setup random hyperparameter search for Logisticregression
    rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                    param_distributions=log_reg_grid,
                                    cv=5,
                                    n_iter=20,
                                    verbose=True)

    # fitting random hyperparameter search model for logisticRegression
    rs_log_reg.fit(X_train, y_train)

    arr = np.array([a, b, c, d, e, f, g, i, j, k, l, m, n])
    query = arr.reshape(1, -1) # Reshape the array
    prediction = rs_log_reg.predict(query)[0]

    return prediction 
