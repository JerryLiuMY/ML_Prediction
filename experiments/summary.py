from global_settings import DATA_PATH
from global_settings import OUTPUT_PATH
from global_settings import cusip_all
from tools.utils import ignore_warnings
import pandas as pd
import glob
import json
import os


@ignore_warnings
def summarize(model_name, window):
    """ Build summary statistics for ML prediction results
    :param model_name: model name
    :param window: [trddt_train, trddt_valid, trddt_test] window
    """

    # find dates in the directory
    window_path = os.path.join(OUTPUT_PATH, model_name, window["name"])
    file_names = [_.split("/")[-1] for _ in glob.glob(os.path.join(window_path, "predict", "*.pkl"))]
    dates = sorted([_.split(".")[0] for _ in file_names])

    # build the summary dataframe
    true_df = pd.DataFrame(index=cusip_all)
    pred_df = pd.DataFrame(index=cusip_all)

    for date in dates:
        true_sub_df = pd.read_pickle(os.path.join(DATA_PATH, "y", f"{date}.pkl"))
        pred_sub_df = pd.read_pickle(os.path.join(window_path, "predict", f"{date}.pkl"))
        true_sub_df = true_sub_df.rename(columns={"target": date})
        pred_sub_df = pred_sub_df.rename(columns={"target": date})
        true_df = true_df.join(true_sub_df, how="left")
        pred_df = pred_df.join(pred_sub_df, how="left")

    # get cross-sectional correlation
    pearson_corr = {}
    spearman_corr = {}
    for date in dates:
        pearson_corr[date] = true_df[date].corr(pred_df[date], method="pearson")
        spearman_corr[date] = true_df[date].corr(pred_df[date], method="spearman")

    with open(os.path.join(window_path, "summary", "pearson_corr.json"), "w") as handle:
        json.dump(pearson_corr, handle)

    with open(os.path.join(window_path, "summary", "spearman_corr.json"), "w") as handle:
        json.dump(spearman_corr, handle)
