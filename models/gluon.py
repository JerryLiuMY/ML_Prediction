import pandas as pd
from autogluon.tabular import TabularPredictor
from global_settings import TEMP_PATH


def fit_autogluon(train_data, valid_data):
    """ Fit autogluon model and report evaluation metric
    :param train_data: dataframe of training data
    :param valid_data: dataframe of validation data
    :return model: fitted model
    """

    # missing data handled automatically
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    predictor = TabularPredictor(label="target", path=TEMP_PATH)
    model = predictor.fit(train_data, tuning_data=valid_data, presets="medium_quality")
    perf = model.evaluate(valid_data, auxiliary_metrics=True, silent=True)
    metric = {"pearsonr": perf["pearsonr"], "RMSE": -perf["root_mean_squared_error"], "r2": perf["r2"]}

    return model, metric


def pre_autogluon(model, test_data):
    """ Make predictions with autogluon model
    :return target: predicted target
    """

    target = pd.DataFrame(model.predict(test_data))

    return target
