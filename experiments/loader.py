import pandas as pd
import numpy as np
from tools.utils import iterable_wrapper
from global_settings import DATA_PATH
import pickle5 as pickle
import os
import torch
num_ft = 798


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

    data_df = pd.DataFrame(columns=[f"x{num}" for num in range(num_ft)] + ["target"])
    for t_X, t_y in zip(trddt_X, trddt_y):
        with open(os.path.join(DATA_PATH, "X", f"{t_X[0]}.pkl"), "rb") as handle:
            X_sub = pickle.load(handle)
        with open(os.path.join(DATA_PATH, "y", f"{t_y}.pkl"), "rb") as handle:
            y_sub = pickle.load(handle)
        data_df_sub = pd.concat([X_sub, y_sub], axis=1, join="inner")
        data_df = pd.concat([data_df, data_df_sub], axis=0)

    return data_df


@iterable_wrapper
def load_dg(trddt_X, trddt_y):
    """ Prepare data generator corresponding to trddt_X and trddt_y
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :return: data generator
    """

    batch_size = 128
    for t_X, t_y in zip(trddt_X, trddt_y):
        X_sub_li = []
        for _ in t_X:
            with open(os.path.join(DATA_PATH, "X", f"{_}.pkl"), "rb") as handle:
                X_sub = pickle.load(handle)
                X_sub_li.append(X_sub)

        with open(os.path.join(DATA_PATH, "y", f"{t_y}.pkl"), "rb") as handle:
            y_sub = pickle.load(handle)

        data_df_sub = pd.concat(X_sub_li, axis=1, join="inner")
        data_df_sub = pd.concat([data_df_sub, y_sub], axis=1, join="inner")
        data_df_sub.dropna(axis=0, how="any", inplace=True)

        for idx in range(0, data_df_sub.shape[0], batch_size):
            data_df_dg = data_df_sub.iloc[idx: idx + batch_size, :]
            X_dg = np.array([data_df_dg.iloc[:, i: i + num_ft].values for i in range(0, num_ft * len(t_X), num_ft)])
            y_dg = data_df_dg["target"].values.reshape(data_df_dg.shape[0], -1)
            X_dg = torch.FloatTensor(X_dg)
            y_dg = torch.FloatTensor(y_dg)
            index = data_df_dg.index

            yield X_dg, y_dg, index
