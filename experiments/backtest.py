import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from global_settings import OUTPUT_PATH
from global_settings import cusip_sic
from params.params import window_dict
import seaborn as sns
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
    corr_ind_df = pd.DataFrame(index=inds, columns=windows).astype("float32")
    for window in windows:
        window_path = os.path.join(horizon_path, str(window))
        with open(os.path.join(window_path, "summary", "corr_ind.json"), "r") as handle:
            corr_ind = json.load(handle)
        for ind in inds:
            corr_ind_df.loc[ind, window] = np.nanmean(list(corr_ind[ind].values()))

    # initialize figure
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(7, 10)
    ax1 = fig.add_subplot(gs[0:5, :])
    ax2 = fig.add_subplot(gs[5:7, :8])

    yticks = np.arange(len(corr_ind_df.index))
    ylabels = [_ if int(_) % 10 == 0 else "" for _ in corr_ind_df.index]
    sns.heatmap(corr_ind_df.values, cmap="YlGnBu", ax=ax1)
    ax1.get_xaxis().set_visible(False)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels)
    ax1.set_ylabel("CIS Industrial Code")

    test_size = window_dict["test_win"] - window_dict["valid_win"]
    xticks = range(len(daily_corr_df.index))
    xlabels = [_ if idx % test_size == 0 else "" for idx, _ in enumerate(daily_corr_df.index)]
    ax2.stem(xticks, daily_corr_df["corr"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax2.scatter(xticks, daily_corr_df["corr"].values, color="#899499", marker=".")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, rotation=30)
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Correlation")
    plt.tight_layout()
