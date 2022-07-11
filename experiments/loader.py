import pandas as pd
from global_settings import DATA_PATH
import pickle5 as pickle
import os


def load_data(trddt_X, trddt_y, data_type):
    """ Load data for experiment
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :param data_type: data type
    """

    if data_type == pd.DataFrame:
        df = load_df(trddt_X, trddt_y)
        return df


def load_df(trddt_X, trddt_y):
    """ Prepare dataframes corresponding to trddt_X and trddt_y
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :return: combined dataframe
    """

    df = pd.DataFrame(columns=[f"x{num}" for num in range(798)] + ["target"])
    for t_X, t_y in zip(trddt_X, trddt_y):
        with open(os.path.join(DATA_PATH, "X", f"{t_X}.pkl"), "rb") as handle:
            X_sub = pickle.load(handle)
        with open(os.path.join(DATA_PATH, "y", f"{t_y}.pkl"), "rb") as handle:
            y_sub = pickle.load(handle)
        df_sub = pd.concat([X_sub, y_sub], axis=1, join="inner")
        df = pd.concat([df, df_sub], axis=0)

    return df
