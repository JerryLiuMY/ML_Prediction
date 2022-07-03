import pandas as pd
import numpy as np
from global_settings import OUTPUT_PATH
from global_settings import cusip_sic
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
sns.set()


def build_correlation(model_name, horizon):
    """ Build industry correlation and daily correlation for each industry
    :param model_name:
    :param horizon:
    :return:
    """

    # define windows and industries
    horizon_path = os.path.join(OUTPUT_PATH, model_name, f"horizon={horizon}")
    windows = sorted([_[1] for _ in os.walk(horizon_path)][0])
    windows = [_ for _ in windows if "correlation" not in _]
    inds = sorted(set([_ for _ in list(cusip_sic["sic"].apply(lambda _: str(_)[:2]))]))

    # industry correlation
    corr_ind_df = pd.DataFrame(index=inds, columns=windows).astype("float32")
    for window in windows:
        window_path = os.path.join(horizon_path, str(window))
        with open(os.path.join(window_path, "summary", "corr_ind.json"), "r") as handle:
            corr_ind = json.load(handle)
        for ind in inds:
            corr_ind_df.loc[ind, window] = np.nanmean(list(corr_ind[ind].values()))

    # daily correlation
    pearson_corr = {}
    spearman_corr = {}
    for window in windows:
        window_path = os.path.join(horizon_path, str(window))
        with open(os.path.join(window_path, "summary", "pearson_corr.json"), "r") as handle:
            pearson_corr.update(json.load(handle))
        with open(os.path.join(window_path, "summary", "spearman_corr.json"), "r") as handle:
            spearman_corr.update(json.load(handle))
    pearson_corr_df = pd.DataFrame.from_dict(pearson_corr, columns=["pearson_corr"], orient="index")
    spearman_corr_df = pd.DataFrame.from_dict(spearman_corr, columns=["spearman_corr"], orient="index")
    corr_df = pd.concat([pearson_corr_df, spearman_corr_df])

    return corr_ind_df, corr_df


def plot_correlation(corr_ind_df, corr_df, horizon_path):
    """ Plot daily & cumulative correlation and heatmap for each industry
    :param corr_ind_df: correlation df for each industry
    :param corr_df: correlation df
    :param horizon_path: horizon path
    :return:
    """

    # initialize correlation plot
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(8, 10)
    ax = fig.add_subplot(gs[0:3, :])
    ax1 = fig.add_subplot(gs[3:4, :8])
    ax2 = fig.add_subplot(gs[3:4, :8])
    ax3 = fig.add_subplot(gs[3:4, :8])
    ax4 = fig.add_subplot(gs[3:4, :8])

    # ax: industrial correlation
    yticks = np.arange(len(corr_ind_df.index))
    ylabels = [_ if int(_) % 10 == 0 else "" for _ in corr_ind_df.index]
    sns.heatmap(corr_ind_df.values, cmap="YlGnBu", ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("SIC Industrial Code")

    # ax1: pearson correlation
    index = range(len(corr_df.index))
    ax1.stem(index, corr_df["pearson_corr"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax1.scatter(index, corr_df["pearson_corr"].values, color="#899499", marker=".")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("Pearson")

    # ax2: cumulative correlation

    # ax3: spearman correlation
    ax3.stem(index, corr_df["spearman_corr"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax3.scatter(index, corr_df["spearman_corr"].values, color="#899499", marker=".")
    ax3.get_xaxis().set_visible(False)
    ax3.set_ylabel("Spearman")

    # ax4: cumulative rank correlation

    # save plotted figure
    plt.tight_layout()
    corr_path = os.path.join(horizon_path, "correlation")
    if not os.path.isdir(corr_path):
        os.mkdir(corr_path)
    fig.savefig(os.path.join(corr_path, "correlation.pdf"), bbox_inches="tight")
