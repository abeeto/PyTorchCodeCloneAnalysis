import os
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    input_path = ""
    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    # create a new column and assign a dummy value (say -1)
    df["kfold"] = -1
    # shuffle the dataset randomly and reset the index to sequential order
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=10)
    for fold_, (train_idx, test_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[test_idx, "kfold"] = fold_
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
