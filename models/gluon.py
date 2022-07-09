from autogluon.tabular import TabularPredictor
import pandas as pd
import os


def fit_autogluon(train_data, valid_data, params, window_path):
    """ Fit autogluon model and report evaluation metric
    :param train_data: dataframe of training data
    :param valid_data: dataframe of validation data
    :param params: dictionary of parameters
    :param window_path: path to the particular window
    :return model: fitted model
    """

    # missing data handled automatically
    presets = params["medium_quality"]
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    predictor = TabularPredictor(label="target", path=os.path.join(window_path, "model"))
    model = predictor.fit(train_data, tuning_data=valid_data, presets=presets,
                          ag_args_fit={"num_gpus": 1}, verbosity=2)
    model.save_space()

    perf = model.evaluate(valid_data, auxiliary_metrics=True, silent=True)
    metric = {"pearsonr": perf["pearsonr"], "RMSE": -perf["root_mean_squared_error"], "r2": perf["r2"]}

    return model, metric


def pre_autogluon(model, test_data):
    """ Make predictions with autogluon model
    :return target: predicted target
    """

    target = pd.DataFrame(model.predict(test_data))

    return target


# def get_leaderboard(model, test_data):
#     """ Get the leaderboard of models in the ensemble
#     :return: leaderboard
#     """
#
#     model = TabularPredictor.load("model_path")
#     leaderboard = model.leaderboard(test_data, silent=True)
#
#     return leaderboard
