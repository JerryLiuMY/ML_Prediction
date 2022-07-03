from global_settings import DATA_PATH
from global_settings import OUTPUT_PATH
from global_settings import cusip_sic
import pandas as pd
import glob
import os


def summarize(model_name, horizon, window):
    """ Build summary statistics for ML prediction results
    :param model_name: model name
    :param horizon: predictive horizon
    :param window: [trddt_train, trddt_valid, trddt_test] window
    """

    # find dates in the directory
    window_path = os.path.join(OUTPUT_PATH, model_name, f"horizon={horizon}", window["X"][0][0])
    file_names = [_.split("/")[-1] for _ in glob.glob(os.path.join(window_path, "predict", "*.pkl"))]
    dates = [_.split(".")[0] for _ in file_names]

    # build the summary dataframe
    true_df = pd.DataFrame(index=cusip_sic["cusip"])
    pred_df = pd.DataFrame(index=cusip_sic["cusip"])

    for date in dates:
        true_sub_df = pd.read_pickle(os.path.join(DATA_PATH, "y", f"{date}.pkl"))
        pred_sub_df = pd.read_pickle(os.path.join(window_path, "predict", f"{date}.pkl"))
        true_sub_df = true_sub_df.rename(columns={"target": date})
        pred_sub_df = pred_sub_df.rename(columns={"target": date})
        true_df = true_df.join(true_sub_df, how="left")
        pred_df = pred_df.join(pred_sub_df, how="left")

    return true_df, pred_df

    # get cross-sectional correlation

    # get time series correlation

    # get cross-sectional correlation for each industry

    # get time series correlation for each industry

    pass
