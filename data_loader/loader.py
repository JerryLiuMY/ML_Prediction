import pandas as pd
import numpy as np
from global_settings import DATA_PATH
import pickle5 as pickle
import os
import torch


def load_data(trddt_X, trddt_y, model_name):
    """ Load data for experiment
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :param model_name: model name
    """

    if model_name == "autogluon":
        data_df = load_df(trddt_X, trddt_y)
        return data_df
    elif model_name == "transformer":
        data_dg = load_dg(trddt_X, trddt_y)
        return data_dg


def load_df(trddt_X, trddt_y):
    """ Prepare dataframes corresponding to trddt_X and trddt_y
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :return: combined dataframe
    """

    data_df = pd.DataFrame(columns=[f"x{num}" for num in range(798)] + ["target"])
    for t_X, t_y in zip(trddt_X, trddt_y):
        with open(os.path.join(DATA_PATH, "X", f"{t_X[0]}.pkl"), "rb") as handle:
            X_sub = pickle.load(handle)
        with open(os.path.join(DATA_PATH, "y", f"{t_y}.pkl"), "rb") as handle:
            y_sub = pickle.load(handle)
        data_df_sub = pd.concat([X_sub, y_sub], axis=1, join="inner")
        data_df = pd.concat([data_df, data_df_sub], axis=0)

    return data_df


def load_dg(trddt_X, trddt_y):
    """ Prepare data generator corresponding to trddt_X and trddt_y
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :return: data generator
    """

    batch_size = 512
    for t_X, t_y in zip(trddt_X, trddt_y):
        data_df_sub = pd.DataFrame(columns=[f"x{num}" for num in range(798)] * len(t_X) + ["target"])
        for _ in t_X:
            with open(os.path.join(DATA_PATH, "X", f"{_}.pkl"), "rb") as handle:
                X_sub = pickle.load(handle)
                data_df_sub = pd.concat([data_df_sub, X_sub], axis=1, join="inner")

        with open(os.path.join(DATA_PATH, "y", f"{t_y}.pkl"), "rb") as handle:
            y_sub = pickle.load(handle)
            data_df_sub = pd.concat([data_df_sub, y_sub], axis=1, join="inner")

        for idx in range(0, data_df_sub.shape[0], batch_size):
            data_df_dg = data_df_sub.iloc[idx: idx + batch_size, :]
            X_dg = np.array([data_df_dg.iloc[:, i: i + 798].values for i in range(0, data_df_sub.shape[1], 798)])
            y_dg = data_df_dg["target"].values.reshape(batch_size, -1)
            X_dg = torch.FloatTensor(X_dg)
            y_dg = torch.FloatTensor(y_dg)

            yield X_dg, y_dg
