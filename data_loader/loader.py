import numpy as np
import pandas as pd
import torch

from global_settings import DATA_PATH
import pickle5 as pickle
import os
from models.trans import seq_len, device


def load_data(trddt_X, trddt_y, data_type):
    """ Load data for experiment
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :param data_type: data type
    """

    if data_type == pd.DataFrame:
        data_df = load_df(trddt_X, trddt_y)
        return data_df


def load_df(trddt_X, trddt_y):
    """ Prepare dataframes corresponding to trddt_X and trddt_y
    :param trddt_X: trading dates for X
    :param trddt_y: trading dates for y
    :return: combined dataframe
    """

    data_df = pd.DataFrame(columns=[f"x{num}" for num in range(798)] + ["target"])
    for t_X, t_y in zip(trddt_X, trddt_y):
        with open(os.path.join(DATA_PATH, "X", f"{t_X}.pkl"), "rb") as handle:
            X_sub = pickle.load(handle)
        with open(os.path.join(DATA_PATH, "y", f"{t_y}.pkl"), "rb") as handle:
            y_sub = pickle.load(handle)
        data_df_sub = pd.concat([X_sub, y_sub], axis=1, join="inner")
        data_df = pd.concat([data_df, data_df_sub], axis=0)

    return data_df


# train_data, val_data = get_data()
# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def get_data():
    # construct a littel toy dataset
    time = np.arange(0, 400, 0.1)
    amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    sampels = 2600
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    train_sequence = create_inout_sequences(train_data, seq_len)
    train_sequence = train_sequence[:-1]

    test_data = create_inout_sequences(test_data, seq_len)
    test_data = test_data[:-1]

    return train_sequence.to(device), test_data.to(device)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + 1:i + tw + 1]
        inout_seq.append((train_seq, train_label))

    return torch.FloatTensor(inout_seq)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(seq_len, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(seq_len, 1))

    return input, target
