import pandas as pd
from experiments.loader import load_data


def experiment(window, model_name):
    """ Run experiment for ML prediction
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param model_name: model name
    :return:
    """

    [trddt_train_X, trddt_valid_X, trddt_test_X] = window["X"]
    [trddt_train_y, trddt_valid_y, trddt_test_y] = window["y"]

    if model_name == "autogluon":
        data_type = pd.DataFrame
    else:
        raise ValueError("Invalid model name")

    train_df = load_data(trddt_train_X, trddt_train_y, data_type)
    valid_df = load_data(trddt_valid_X, trddt_valid_y, data_type)
    test_df = load_data(trddt_test_X, trddt_test_y, data_type)
