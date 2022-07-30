from global_settings import OUTPUT_PATH
import matplotlib.pyplot as plt
from params.params import data_dict
import matplotlib.patches as mpatches
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
sns.set()


def build_correlation(model_name):
    """ Build industry correlation and daily correlation for each industry
    :param model_name: model name
    :return:
    """

    # define windows and industries
    model_path = os.path.join(OUTPUT_PATH, model_name)
    test_size = data_dict["test_win"] - data_dict["valid_win"]
    windows = [_[1] for _ in os.walk(model_path)][0]
    windows = sorted([str(_) for _ in windows])
    windows = [_ for _ in windows if "-" in _]

    # daily correlation
    pearson_corr = {}
    spearman_corr = {}
    for window in windows:
        window_path = os.path.join(model_path, str(window))
        with open(os.path.join(window_path, "summary", "pearson_corr.json"), "r") as handle:
            pearson_corr.update(json.load(handle))
        with open(os.path.join(window_path, "summary", "spearman_corr.json"), "r") as handle:
            spearman_corr.update(json.load(handle))
    pearson_df = pd.DataFrame.from_dict(pearson_corr, columns=["pearson"], orient="index")
    spearman_df = pd.DataFrame.from_dict(spearman_corr, columns=["spearman"], orient="index")
    corr_df = pd.concat([pearson_df, spearman_df], axis=1)
    corr_df = corr_df.fillna(0, inplace=False)

    # decay correlation
    decay_df = pd.DataFrame(index=range(0, test_size))
    for i in range(0, corr_df.shape[0], test_size):
        decay_df_sub = corr_df[["pearson"]].iloc[i:i + test_size].reset_index(drop=True)
        decay_df_sub.columns = corr_df[["pearson"]].iloc[[i], :].index
        decay_df = pd.concat([decay_df, decay_df_sub], axis=1)
    decay_df = decay_df.groupby(np.arange(decay_df.shape[0]) // 6).mean()

    return decay_df, corr_df


def plot_correlation(model_name):
    """ Plot daily & cumulative correlation and heatmap for each industry
    :param model_name: model name
    :return:
    """

    # get predictive horizon
    decay_df, corr_df = build_correlation(model_name)
    model_path = os.path.join(OUTPUT_PATH, model_name)
    params_path = os.path.join(model_path, "params")
    with open(os.path.join(params_path, "horizon.json"), "r") as handle:
        horizon_dict = json.load(handle)
        horizon = horizon_dict["horizon"]

    # initialize correlation plot
    test_size = data_dict["test_win"] - data_dict["valid_win"]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(5, 10)
    axa = fig.add_subplot(gs[0:1, :])
    ax1 = fig.add_subplot(gs[1:2, :8])
    ax2 = fig.add_subplot(gs[2:3, :8])
    ax3 = fig.add_subplot(gs[3:4, :8])
    ax4 = fig.add_subplot(gs[4:5, :8])

    # axa: correlation over times
    yticks = np.arange(len(decay_df.index))
    ylabels = [_ * 6 if int(_) % 2 == 0 else "" for _ in yticks]
    sns.heatmap(decay_df.values, cmap="coolwarm", ax=axa)
    axa.get_xaxis().set_visible(False)
    axa.set_yticks(yticks)
    axa.set_yticklabels(ylabels, rotation=0)
    axa.set_ylabel("Decay")
    axa.set_title(f"Model {model_name} with horizon={horizon}" + "\n", fontsize=13)

    # define indices
    index = range(len(corr_df.index))
    xticks = [idx for idx, _ in enumerate(corr_df.index) if idx % (test_size * 2) == 0]
    xlabels = [_ for idx, _ in enumerate(corr_df.index) if idx % (test_size * 2) == 0]
    # xticks = xticks + [len(corr_df.index) - 1]
    # xlabels = xlabels + [corr_df.index[-1]]

    # ax1: pearson correlation
    pearson_mean = round(corr_df["pearson"].mean() * 100, 2)
    pearson_sharpe = round(corr_df["pearson"].mean() / corr_df["pearson"].std() * np.sqrt(252), 3)
    pearson_acf_l1 = round(sm.tsa.acf(corr_df["pearson"].values)[1], 3)
    ax1_legend_mean = mpatches.Patch(color="#A9A9A9", label=f"ave={pearson_mean:.2f}%")
    ax1_legend_sharpe = mpatches.Patch(color="#A9A9A9", label=f"sharpe={pearson_sharpe:.3f}")
    ax1_legend_acf_l1 = mpatches.Patch(color="#A9A9A9", label=f"acf_l1={pearson_acf_l1:.3f}")
    handles = [ax1_legend_mean, ax1_legend_sharpe, ax1_legend_acf_l1]

    ax1.stem(index, corr_df["pearson"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax1.scatter(index, corr_df["pearson"].values, color="#899499", marker=".")
    ax1.legend(handles=handles, loc="upper left", handlelength=0.2, handletextpad=0.5)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Pearson")

    # ax2: spearman correlation
    spearman_mean = round(corr_df["spearman"].mean() * 100, 2)
    spearman_sharpe = round(corr_df["spearman"].mean() / corr_df["spearman"].std() * np.sqrt(252), 3)
    spearman_acf_l1 = round(sm.tsa.acf(corr_df["spearman"].values)[1], 3)
    ax2_legend_mean = mpatches.Patch(color="#A9A9A9", label=f"ave={spearman_mean:.2f}%")
    ax2_legend_sharpe = mpatches.Patch(color="#A9A9A9", label=f"sharpe={spearman_sharpe:.3f}")
    ax2_legend_acf_l1 = mpatches.Patch(color="#A9A9A9", label=f"acf_l1={spearman_acf_l1:.3f}")
    handles = [ax2_legend_mean, ax2_legend_sharpe, ax2_legend_acf_l1]

    ax2.stem(index, corr_df["spearman"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax2.scatter(index, corr_df["spearman"].values, color="#899499", marker=".")
    ax2.legend(handles=handles, loc="upper left", handlelength=0.2, handletextpad=0.5)
    ax2.set_xticklabels([])
    ax2.set_ylabel("Spearman")

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
    model_path = os.path.join(OUTPUT_PATH, model_name)
    corr_path = os.path.join(model_path, "correlation")
    if not os.path.isdir(corr_path):
        os.mkdir(corr_path)
        fig.savefig(os.path.join(corr_path, "correlation.pdf"), bbox_inches="tight")
