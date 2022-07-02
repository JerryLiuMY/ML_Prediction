from autogluon.tabular import TabularPredictor


def fit_autogluon(train_data, valid_data):
    """ Fit autogluon model
    :param train_data: dataframe of training data
    :param valid_data: dataframe of validation data
    :return model: fitted model
    """

    # missing data handled automatically
    model = TabularPredictor(label="target").fit(train_data, tuning_data=valid_data, presets="medium_quality")
    y_pred = model.predict(valid_data)
    perf = model.evaluate_predictions(y_true=valid_data["target"], y_pred=y_pred, auxiliary_metrics=True, silent=True)

    return model, perf


def pre_autogluon(model, test_data):
    """ Make predictions with autogluon model
    :return target: predicted target
    """

    target = model.predict(test_data)

    return target
