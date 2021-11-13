import pandas as pd


def transform_data(data_file):
    cv_df = pd.read_csv(data_file, ',')
    return cv_df
