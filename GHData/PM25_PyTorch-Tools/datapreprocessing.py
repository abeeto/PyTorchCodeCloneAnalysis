#%%
from utils import *

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype


def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


# target is one of the ["is_canceled", "reservation_status", "adr"]
def processing_data(target="is_canceled", fname="data/train.csv", use_dummies=True):
    train_df = pd.read_csv(fname)

    train_df["expected_room"] = 0
    train_df.loc[
        train_df["reserved_room_type"] == train_df["assigned_room_type"],
        "expected_room",
    ] = 1
    train_df["net_cancelled"] = 0
    train_df.loc[
        train_df["previous_cancellations"] > train_df["previous_bookings_not_canceled"],
        "net_cancelled",
    ] = 1

    # TrainDataVisualization(train_df, None,).correlation_matrix().show()

    exclude_columns = [
        target,
        "is_canceled",
        "ID",
        "adr",
        "reservation_status",
        "reservation_status_date",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "arrival_date_month",
        "assigned_room_type",
        "reserved_room_type",
        "previous_cancellations",
        "previous_bookings_not_canceled",
    ]

    if is_numeric_dtype(train_df[target]):
        y_df = train_df[target]
    else:
        y_df = train_df[target].astype("category")
        reservation_status_cats = y_df.cat.categories
        y_df = y_df.cat.codes  # convert categories data to numeric codes

    X_df = train_df.drop(exclude_columns, axis=1)
    X_df.children = X_df.children.fillna(0)
    nan_cols = list(get_columns_with_nan(X_df))
    print(f"Columns that contain NaN: {nan_cols}")

    for col in nan_cols:
        X_df[col] = X_df[col].fillna("Null").astype(str)

    if use_dummies:
        X_df = pd.get_dummies(X_df)
    else:
        for col in X_df.select_dtypes(include=["object"]).columns:
            X_df[col] = X_df[col].factorize()[0]

    print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")

    return (X_df, y_df)


if __name__ == "__main__":
    X, y = processing_data("adr")
