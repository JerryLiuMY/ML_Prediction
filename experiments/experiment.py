from data_loader.loader import load_data
from models.gluon import fit_autogluon, pre_autogluon
from global_settings import OUTPUT_PATH
from datetime import datetime
import pandas as pd
import pickle
import json
import os


def experiment(model_name, horizon, params, window):
    """ Run experiment for ML prediction
    :param model_name: model name
    :param horizon: predictive horizon
    :param params: dictionary of parameters
    :param window: [trddt_train, trddt_valid, trddt_test] window
    """

    # define data_type, fit_func and pre_func
    if model_name == "autogluon":
        data_type = pd.DataFrame
        fit_func = fit_autogluon
        pre_func = pre_autogluon
    else:
        raise ValueError("Invalid model name")

    # get trddt and window_path
    name = window["name"]
    [trddt_train_X, trddt_valid_X, trddt_test_X] = window["X"]
    [trddt_train_y, trddt_valid_y, trddt_test_y] = window["y"]
    model_path = os.path.join(OUTPUT_PATH, model_name)
    window_path = os.path.join(model_path, name)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on {model_name} "
          f"with horizon={horizon} and window={name}")

    # train model with validation
    with open(os.path.join(window_path, "info", "window.pkl"), "wb") as handle:
        pickle.dump(window, handle)

    train_data = load_data(trddt_train_X, trddt_train_y, data_type)
    valid_data = load_data(trddt_valid_X, trddt_valid_y, data_type)

    print(f"trddt_train_X: {trddt_train_X}"), print(f"trddt_train_y: {trddt_train_y}")
    print(f"trddt_valid_X: {trddt_valid_X}"), print(f"trddt_valid_y: {trddt_valid_y}")

    model, metric = fit_func(train_data, valid_data, params, window_path)
    with open(os.path.join(window_path, "info", "metric.json"), "w") as handle:
        json.dump(metric, handle)

    # make predictions
    for t_test_X, t_test_y in zip(trddt_test_X, trddt_test_y):
        print(f"t_test_X: {t_test_X}, t_test_y: {t_test_y}")
        test_data = load_data([t_test_X], [t_test_y], data_type)
        target = pre_func(model, test_data)
        target.index.name = "cusip"
        target.to_pickle(os.path.join(window_path, "predict", f"{t_test_y}.pkl"))
