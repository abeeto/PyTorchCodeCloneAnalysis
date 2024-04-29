import pandas as pd

from filtering_functions import _wrong_data_filtering, _removing_duplicates, _interpolated_data, \
                                _create_clean_data_frame, _pandas_to_numeric, _dropping_nan_columns, _dropping_nan_rows


def filtering(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = _wrong_data_filtering(data_frame)
    data_frame = _removing_duplicates(data_frame)
    data_frame = _interpolated_data(data_frame)
    data_frame = _create_clean_data_frame(data_frame)
    data_frame = _pandas_to_numeric(data_frame)
    _dropping_nan_columns(data_frame)
    _dropping_nan_rows(data_frame)

    return data_frame
