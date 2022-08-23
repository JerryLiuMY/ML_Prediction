from sklearn.linear_model import LinearRegression
import pickle5 as pickle
import pandas as pd
import os


def fit_linear(train_data, valid_data, params, window_path):
    """ Fit autogluon model and report evaluation metric
    :param train_data: dataframe of training data
    :param valid_data: dataframe of validation data
    :param params: dictionary of parameters
    :param window_path: path to the particular window
    :return model: fitted model
    """

    # columns
    fit_intercept = params["fit_intercept"]
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    X = train_data.loc[:, train_data.columns != "target"].values
    y = train_data.loc[:, "target"].values
    predictor = LinearRegression(fit_intercept=fit_intercept)

    # fit the predictors and perform evaluation
    model = predictor.fit(X, y)
    save_path = os.path.join(window_path, "model")
    with open(os.path.join(save_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    X_valid = valid_data.loc[:, valid_data.columns != "target"].values
    y_valid = valid_data.loc[:, "target"].values
    score = model.score(X_valid, y_valid)
    metric = {"score": score}

    return model, metric


def pre_linear(model, test_data):
    """ Make predictions with autogluon model
    :param model: fitted model
    :param test_data: dataframe of testing data
    :return target: predicted target
    """

    X_test = test_data.loc[:, test_data.columns != "target"].values
    target = pd.DataFrame(data=model.predict(X_test), index=test_data.index, columns=["target"])

    return target
