from typing import Optional
import numpy as np
import pandas as pd


def _func(value: Optional[float]) -> Optional[float]:
    if value == ' ':  # Turning a ' ' (space) char into 'nan' (so afterwards it could be aggregated)
        return pd.to_numeric(value, errors='coerce')
    if (pd.to_numeric(value, errors='coerce') > -np.inf) and int(value) >= 1e3:
        return int(value) / 1e1
    return value


def _wrong_data_filtering(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Wrong Data; Correcting all misleading data by a Decade (log scale - log_10(P) = x => P=1ex)
    # Turning all non-numeric values into 'nan'
    data_frame = data_frame.applymap(lambda value: _func(value))

    return data_frame


def _removing_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Removing Duplicates
    data_frame.drop_duplicates(inplace=True)

    return data_frame


def _interpolated_data(data_frame) -> pd.DataFrame:
    # Replacing 'nan' values with the desired interpolation
    clean_data_frame: pd.DataFrame = data_frame.copy()
    # Changes all the values of the non-numeric cells to 'nan. The result is pandas DataFrame
    clean_data_frame = clean_data_frame.apply(lambda series: pd.to_numeric(series, errors='coerce'))
    # Applying an aggregation function on the DataFrame. The result is pandas series - size: (number_of_columns,)
    interpolation_data_series: pd.DataFrame = clean_data_frame.aggregate(func=np.mean, axis=0)
    # Filling the 'nan' cells with the resulted pandas series above (Because the series above and the DataFrame are the
    # same size, the appliance will be done by matching each value in the 'interpolation_data_series' on it's
    # corresponding series (column) in the DataFrame
    data_frame.fillna(values=interpolation_data_series, inplace=True)

    return data_frame


def _create_clean_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Create clean DataFrame
    drop_clean_set: set = {'-', '_', '/', '\\'}
    clean_data_frame = data_frame.copy()
    for column in clean_data_frame.keys():
        for index, cell in enumerate(clean_data_frame[column]):
            cell = str(cell)
            cell = list(cell)
            cell = [ch for ch in cell if ch not in drop_clean_set]

            # Joining back into string
            s: str = ''
            for ch in cell:
                s += ch
            cell = s
            clean_data_frame.loc[index, column] = cell

    clean_data_frame.replace('nan', np.nan, inplace=True)
    return clean_data_frame


def _pandas_to_numeric(data_frame: pd.DataFrame) -> pd.DataFrame:
    for col in data_frame.columns:
        for index, cell in enumerate(data_frame[col]):
            numeric_value = pd.to_numeric(cell, errors='coerce')
            if numeric_value > -np.inf:  # Is numeric
                data_frame.loc[index, col] = float(numeric_value)

    return data_frame


def _dropping_nan_columns(data_frame: pd.DataFrame) -> None:
    # Dropping NAN Columns
    data_frame.dropna(axis=1, how='all', inplace=True)
    data_frame.reset_index(drop=True, inplace=True)


def _dropping_nan_rows(data_frame: pd.DataFrame) -> None:
    # Dropping NAN Rows
    data_frame.dropna(axis=0, how='all', inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
