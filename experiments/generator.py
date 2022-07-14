from global_settings import trddt_all
import numpy as np


def generate_window(window_dict, date0_min, date0_max, horizon):
    """ generate rolling windows for a set of experiments
    :param window_dict: dictionary of window related parameters
    :param date0_min: earliest date in the enriched data
    :param date0_max: latest date in the enriched data
    :param horizon: predictive horizon
    """

    trddt = trddt_all[(trddt_all >= date0_min) & (trddt_all <= date0_max)].tolist()

    train_win = window_dict["train_win"]
    valid_win = window_dict["valid_win"]
    test_win = window_dict["test_win"]
    resample = window_dict["resample"]

    for i in range(0, len(trddt) - test_win - horizon + 1, test_win - valid_win):
        trddt_train_X = trddt[i: i + train_win]
        trddt_valid_X = trddt[i + train_win: i + valid_win]
        trddt_test_X = trddt[i + valid_win: i + test_win]
        trddt_train_y = trddt[i + horizon: i + horizon + train_win]
        trddt_valid_y = trddt[i + horizon + train_win: i + horizon + valid_win]
        trddt_test_y = trddt[i + horizon + valid_win: i + horizon + test_win]
        trddt_name = trddt_train_X[0]

        if resample:
            np.random.seed(0)
            trddt_X = trddt_train_X + trddt_valid_X
            trddt_y = trddt_train_y + trddt_valid_y
            index = list(range(len(trddt_X)))
            train_idx = sorted(np.random.choice(index, len(trddt_train_X), replace=False))
            valid_idx = sorted(set(index) - set(train_idx))
            trddt_train_X, trddt_train_y = [trddt_X[_] for _ in train_idx], [trddt_y[_] for _ in train_idx]
            trddt_valid_X, trddt_valid_y = [trddt_X[_] for _ in valid_idx], [trddt_y[_] for _ in valid_idx]

        yield {"name": trddt_name,
               "X": [trddt_train_X, trddt_valid_X, trddt_test_X],
               "y": [trddt_train_y, trddt_valid_y, trddt_test_y]}
