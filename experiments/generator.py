from global_settings import trddt_all
import numpy as np


def generate_window(window_dict, date0_min, date0_max, seq_len, horizon):
    """ generate rolling windows for a set of experiments
    :param window_dict: dictionary of window related parameters
    :param date0_min: earliest date in the enriched data
    :param date0_max: latest date in the enriched data
    :param seq_len: sequence length
    :param horizon: predictive horizon
    """

    # build rolling chunks
    trddt = trddt_all.tolist()
    trddt_roll_X = np.array([trddt[i: i + seq_len] for i in range(len(trddt) - (seq_len - 1))])
    trddt_roll_y = np.array(trddt[(seq_len - 1):])
    roll_index = tuple([(trddt_roll_y >= date0_min) & (trddt_roll_y <= date0_max)])
    trddt_roll_X = trddt_roll_X[roll_index].tolist()
    trddt_roll_y = trddt_roll_y[roll_index].tolist()

    # fetch hyper-parameters
    train_win = window_dict["train_win"]
    valid_win = window_dict["valid_win"]
    test_win = window_dict["test_win"]
    resample = window_dict["resample"]
    shift = horizon - 1

    # X six days ahead shall not be used
    for i in range(0, len(trddt_roll_X) - test_win - shift + 1, test_win - valid_win):
        trddt_train_X = trddt_roll_X[i: i + train_win]
        trddt_valid_X = trddt_roll_X[i + train_win: i + valid_win - 6]
        trddt_test_X = trddt_roll_X[i + valid_win: i + test_win]
        trddt_train_y = trddt_roll_y[i + shift: i + shift + train_win]
        trddt_valid_y = trddt_roll_y[i + shift + train_win: i + shift + valid_win - 6]
        trddt_test_y = trddt_roll_y[i + shift + valid_win: i + shift + test_win]
        trddt_name = trddt_roll_y[i]

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
