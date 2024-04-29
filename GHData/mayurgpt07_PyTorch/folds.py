import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('./imdb.csv')

    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    df['kfold'] = -1
    
    y = df.sentiment.values

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for f, (t_, v_) in enumerate(kf.split(X = df, y = y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv('./imdb_folds.csv', index = False)