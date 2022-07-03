import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from global_settings import OUTPUT_PATH
from global_settings import cusip_sic
import json
import os


def plot_correlation(model_name, horizon):
    """ Plot cumulative correlation and heatmap for each industry
    :param model_name:
    :param horizon:
    :return:
    """

    # define windows and industries
    horizon_path = os.path.join(OUTPUT_PATH, model_name, f"horizon={horizon}")
    windows = sorted([_[1] for _ in os.walk(horizon_path)][0])
    inds = sorted(set([_ for _ in list(cusip_sic["sic"].apply(lambda _: str(_)[:2]))]))

    # daily correlation
    daily_corr = {}
    for window in windows:
        window_path = os.path.join(horizon_path, str(window))
        with open(os.path.join(window_path, "summary", "daily_corr.json"), "r") as handle:
            daily_corr.update(json.load(handle))
    daily_corr_df = pd.DataFrame.from_dict(daily_corr, columns=["corr"], orient="index")

    # industry correlation
    corr_ind_df = pd.DataFrame(index=inds, columns=windows)
    for window in windows:
        window_path = os.path.join(horizon_path, str(window))
        with open(os.path.join(window_path, "summary", "corr_ind.json"), "r") as handle:
            corr_ind = json.load(handle)
        for ind in inds:
            corr_ind_df.loc[ind, window] = np.nanmean(list(corr_ind[ind].values()))

    return daily_corr_df, corr_ind_df
