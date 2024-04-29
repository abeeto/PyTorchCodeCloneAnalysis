import numpy as np
import pandas as pd
from copy import deepcopy
from prepare_datasets import get_iemocap_raw


# [x_train, x_val, x_test, y_train, y_val, y_test] = get_iemocap_raw()

def balance_classes(x, y):
    df = pd.DataFrame(data=x)
    df['y'] = y
    x_new, y_new = x, y
    classes = np.unique(y, return_counts=True)[0]
    classes_count = np.unique(y, return_counts=True)[1]
    max_number = np.max(classes_count)
    for idx, cls in enumerate(classes):
        mask = np.random.choice(classes_count[idx], size=max_number-classes_count[idx], replace=True)
        sub_df = df[df.y == cls]
        x_to_add = sub_df.drop(columns=['y']).values[mask,:]
        x_new = np.vstack((x_new, x_to_add))

        y_new = np.hstack((y_new, np.full((len(mask),), cls)))


    return np.array(x_new), np.array(y_new)

# x_new, y_new = balance_classes(x_train, y_train)




