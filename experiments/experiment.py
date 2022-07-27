from experiments.loader import load_data
from models.gluon import fit_autogluon, pre_autogluon
from models.trans import fit_transformer, pre_transformer
from global_settings import OUTPUT_PATH
from datetime import datetime
import pickle5 as pickle
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
        fit_func = fit_autogluon
        pre_func = pre_autogluon
    elif model_name == "transformer":
        fit_func = fit_transformer
        pre_func = pre_transformer
    else:
        raise ValueError("Invalid model name")

    # get window_path and trddt
    name = window["name"]
    model_path = os.path.join(OUTPUT_PATH, model_name)
    window_path = os.path.join(model_path, name)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on {model_name} "
          f"with horizon={horizon} and window={name}")

    [trddt_train_X, trddt_valid_X, trddt_test_X] = window["X"]
    [trddt_train_y, trddt_valid_y, trddt_test_y] = window["y"]
    print(f"trddt_train_X: {trddt_train_X}"), print(f"trddt_train_y: {trddt_train_y}")
    print(f"trddt_valid_X: {trddt_valid_X}"), print(f"trddt_valid_y: {trddt_valid_y}")

    # train model with validation
    with open(os.path.join(window_path, "info", "window.pkl"), "wb") as handle:
        pickle.dump(window, handle, protocol=4)

    train_data = load_data(trddt_train_X, trddt_train_y, model_name)
    valid_data = load_data(trddt_valid_X, trddt_valid_y, model_name)
    model, metric = fit_func(train_data, valid_data, params, window_path)
    with open(os.path.join(window_path, "info", "metric.json"), "w") as handle:
        json.dump(metric, handle)

    # make predictions
    for t_test_X, t_test_y in zip(trddt_test_X, trddt_test_y):
        print(f"t_test_X: {t_test_X}, t_test_y: {t_test_y}")
        test_data = load_data([t_test_X], [t_test_y], model_name)
        target = pre_func(model, test_data)
        target.index.name = "cusip"
        target.to_pickle(os.path.join(window_path, "predict", f"{t_test_y}.pkl"), protocol=4)
