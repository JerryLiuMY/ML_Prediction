from global_settings import OUTPUT_PATH
from global_settings import cusip_sic
import matplotlib.pyplot as plt
from params.params import window_dict
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
sns.set()


def build_correlation(model_name, horizon):
    """ Build industry correlation and daily correlation for each industry
    :param model_name: model name
    :param horizon: predictive horizon
    :return:
    """

    # define windows and industries
    horizon_path = os.path.join(OUTPUT_PATH, model_name, f"horizon={horizon}")
    test_size = window_dict["test_win"] - window_dict["valid_win"]
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
    pearson_df = pd.DataFrame.from_dict(pearson_corr, columns=["pearson"], orient="index")
    spearman_df = pd.DataFrame.from_dict(spearman_corr, columns=["spearman"], orient="index")
    corr_df = pd.concat([pearson_df, spearman_df], axis=1)

    # decay correlation
    decay_df = pd.DataFrame(index=range(0, test_size))
    for i in range(0, corr_df.shape[0], test_size):
        decay_df_sub = corr_df[["pearson"]].iloc[i:i + test_size].reset_index(drop=True)
        decay_df_sub.columns = corr_df[["pearson"]].iloc[[i], :].index
        decay_df = pd.concat([decay_df, decay_df_sub], axis=1)
    decay_df = decay_df.groupby(np.arange(decay_df.shape[0]) // 6).mean()

    return corr_ind_df, decay_df, corr_df


def plot_correlation(model_name, horizon):
    """ Plot daily & cumulative correlation and heatmap for each industry
    :param model_name: model name
    :param horizon: predictive horizon
    :return:
    """

    # build and filter correlation results
    corr_ind_df, decay_df, corr_df = build_correlation(model_name, horizon)
    corr_ind_df = corr_ind_df.loc[:, corr_ind_df.apply(lambda _: _.name[:4] < "2020")]
    decay_df = decay_df.loc[:, decay_df.apply(lambda _: _.name[:4] < "2020")]
    corr_df = corr_df.loc[corr_df.apply(lambda _: _.name[:4] < "2020", axis=1), :]

    # initialize correlation plot
    test_size = window_dict["test_win"] - window_dict["valid_win"]
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(9, 10)
    axa = fig.add_subplot(gs[0:4, :])
    axb = fig.add_subplot(gs[4:5, :])
    ax1 = fig.add_subplot(gs[5:6, :8])
    ax2 = fig.add_subplot(gs[6:7, :8])
    ax3 = fig.add_subplot(gs[7:8, :8])
    ax4 = fig.add_subplot(gs[8:9, :8])

    # axa: industrial correlation
    yticks = np.arange(len(corr_ind_df.index))
    ylabels = [_ if int(_) % 10 == 0 else "" for _ in corr_ind_df.index]
    sns.heatmap(corr_ind_df.values, cmap="YlGnBu", ax=axa)
    axa.get_xaxis().set_visible(False)
    axa.set_yticks(yticks)
    axa.set_yticklabels(ylabels)
    axa.set_ylabel("SIC Industrial Code")
    axa.set_title(f"Model {model_name} with horizon={horizon}" + "\n", fontsize=13)

    # axb: correlation over times
    yticks = np.arange(len(decay_df.index))
    ylabels = [_ * 6 if int(_) % 2 == 0 else "" for _ in yticks]
    sns.heatmap(decay_df.values, cmap="coolwarm", ax=axb)
    axb.get_xaxis().set_visible(False)
    axb.set_yticks(yticks)
    axb.set_yticklabels(ylabels, rotation=0)
    axb.set_ylabel("Decay")

    # define indices
    index = range(len(corr_df.index))
    xticks = [idx for idx, _ in enumerate(corr_df.index) if idx % (test_size * 2) == 0]
    xlabels = [_ for idx, _ in enumerate(corr_df.index) if idx % (test_size * 2) == 0]
    xticks = xticks + [len(corr_df.index) - 1]
    xlabels = xlabels + [corr_df.index[-1]]

    # ax1: pearson correlation
    pearson_mean = round(corr_df["pearson"].mean() * 100, 2)
    pearson_sharpe = round(corr_df["pearson"].mean() / corr_df["pearson"].std() * np.sqrt(252), 3)
    ax1_legend_mean = mpatches.Patch(color="#A9A9A9", label=f"ave={pearson_mean:.2f}%")
    ax1_legend_sharpe = mpatches.Patch(color="#A9A9A9", label=f"sharpe={pearson_sharpe:.3f}")
    ax1.stem(index, corr_df["pearson"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax1.scatter(index, corr_df["pearson"].values, color="#899499", marker=".")
    ax1.legend(handles=[ax1_legend_mean, ax1_legend_sharpe], loc="upper left", handlelength=0.2, handletextpad=0.5)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Pearson")
    ax1.set_ylim([-0.30, 0.40])

    # ax2: spearman correlation
    spearman_mean = round(corr_df["spearman"].mean() * 100, 2)
    spearman_sharpe = round(corr_df["spearman"].mean() / corr_df["spearman"].std() * np.sqrt(252), 3)
    ax2_legend_mean = mpatches.Patch(color="#A9A9A9", label=f"ave={spearman_mean:.2f}%")
    ax2_legend_sharpe = mpatches.Patch(color="#A9A9A9", label=f"sharpe={spearman_sharpe:.3f}")
    ax2.stem(index, corr_df["spearman"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax2.scatter(index, corr_df["spearman"].values, color="#899499", marker=".")
    ax2.legend(handles=[ax2_legend_mean, ax2_legend_sharpe], loc="upper left", handlelength=0.2, handletextpad=0.5)
    ax2.set_xticklabels([])
    ax2.set_ylabel("Spearman")
    ax2.set_ylim([-0.25, 0.25])

    # ax3: cumulative spearman correlation
    ax3.plot(index, np.log(np.cumprod(corr_df["pearson"].values + 1)), color="#6E6E6E", label="pearson")
    ax3.plot(index, np.log(np.cumprod(corr_df["spearman"].values + 1)), color="#808080", label="spearman")
    ax3.legend(loc="upper left")
    ax3.set_xticklabels([])
    ax3.set_ylabel("log(1+corr)")
    ax3.grid(True)

    # ax4: cumulative spearman correlation
    ax4.plot(index, np.cumsum(corr_df["pearson"].values), color="#6E6E6E", label="pearson")
    ax4.plot(index, np.cumsum(corr_df["spearman"].values), color="#808080", label="spearman")
    ax4.legend(loc="upper left")
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xlabels, rotation=25)
    ax4.set_ylabel("sum(corr)")
    ax4.grid(True)

    # save plotted figure
    horizon_path = os.path.join(OUTPUT_PATH, model_name, f"horizon={horizon}")
    corr_path = os.path.join(horizon_path, "correlation")
    if not os.path.isdir(corr_path):
        os.mkdir(corr_path)
    fig.savefig(os.path.join(corr_path, "correlation.pdf"), bbox_inches="tight")
