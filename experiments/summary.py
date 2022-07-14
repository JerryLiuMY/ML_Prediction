from global_settings import DATA_PATH
from global_settings import OUTPUT_PATH
from global_settings import cusip_sic
from tools.utils import ignore_warnings
import pickle5 as pickle
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
    industry = list(cusip_sic["sic"].apply(lambda _: int(str(_)[:2])))
    true_df = pd.DataFrame(data={"industry": industry}, index=cusip_sic["cusip"])
    pred_df = pd.DataFrame(data={"industry": industry}, index=cusip_sic["cusip"])

    for date in dates:
        with open(os.path.join(DATA_PATH, "y", f"{date}.pkl"), "rb") as handle:
            true_sub_df = pickle.load(handle)
        with open(os.path.join(window_path, "predict", f"{date}.pkl"), "rb") as handle:
            pred_sub_df = pickle.load(handle)
        true_sub_df = true_sub_df.rename(columns={"target": date})
        pred_sub_df = pred_sub_df.rename(columns={"target": date})
        true_df = true_df.join(true_sub_df, how="left")
        pred_df = pred_df.join(pred_sub_df, how="left")

    # get cross-sectional correlation for each industry
    daily_corr_ind = {}
    for ind in sorted(set(industry)):
        daily_corr_temp = {}
        true_df_ind = true_df.loc[true_df["industry"] == ind]
        pred_df_ind = pred_df.loc[pred_df["industry"] == ind]
        for date in dates:
            daily_corr_temp[date] = true_df_ind[date].corr(pred_df_ind[date], method="pearson")

        daily_corr_ind[str(ind)] = daily_corr_temp

    # get cross-sectional correlation
    pearson_corr = {}
    spearman_corr = {}
    for date in dates:
        pearson_corr[date] = true_df[date].corr(pred_df[date], method="pearson")
        spearman_corr[date] = true_df[date].corr(pred_df[date], method="spearman")

    # get time-series correlation
    cusip_corr = {}
    for cusip in cusip_sic["cusip"].values:
        true_df_row = true_df.loc[cusip, true_df.columns != "industry"]
        pred_df_row = pred_df.loc[cusip, pred_df.columns != "industry"]
        cusip_corr[cusip] = true_df_row.corr(pred_df_row, method="pearson")

    with open(os.path.join(window_path, "summary", "corr_ind.json"), "w") as handle:
        json.dump(daily_corr_ind, handle)

    with open(os.path.join(window_path, "summary", "pearson_corr.json"), "w") as handle:
        json.dump(pearson_corr, handle)

    with open(os.path.join(window_path, "summary", "spearman_corr.json"), "w") as handle:
        json.dump(spearman_corr, handle)

    with open(os.path.join(window_path, "summary", "cusip_corr.json"), "w") as handle:
        json.dump(cusip_corr, handle)
