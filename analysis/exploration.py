import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from global_settings import DATA_PATH
from global_settings import LOG_PATH
from tqdm import tqdm_notebook
import os
sns.set()


def build_count(trddt_all):
    """ Count for the number of unique cusip in the X and y dataframes """

    # build count_df
    count_df = pd.DataFrame(columns=["X", "y", "common"], index=trddt_all)
    for date in tqdm_notebook(trddt_all):
        X_cusip = list(pd.read_pickle(os.path.join(DATA_PATH, "X", f"{date}.pkl")).index)
        y_cusip = list(pd.read_pickle(os.path.join(DATA_PATH, "y", f"{date}.pkl")).index)
        common_cusip = list(set(X_cusip) & set(y_cusip))
        count_df.loc[date, "X"] = len(X_cusip)
        count_df.loc[date, "y"] = len(y_cusip)
        count_df.loc[date, "common"] = len(common_cusip)
    count_df.index.name = "date"
    count_df.reset_index(drop=False, inplace=True)
    count_df.to_csv(os.path.join(LOG_PATH, "count.csv"))


def build_missing(trddt_all):
    """ Build percentage for missing data in the X variable """

    # build missing_df
    missing_df = pd.DataFrame(columns=["stock_perc", "missing_perc"], index=trddt_all)
    for date in tqdm_notebook(trddt_all):
        X = pd.read_pickle(os.path.join(DATA_PATH, "X", f"{date}.pkl"))
        missing_df.loc[date, "stock_perc"] = X[X.isnull().any(axis=1)].shape[0] / X.shape[0]
        missing_df.loc[date, "overall_perc"] = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
    missing_df.index.name = "date"
    missing_df.reset_index(drop=False, inplace=True)
    missing_df.to_csv(os.path.join(LOG_PATH, "missing.csv"))


def plot_exploration(count_df, missing_df):
    """ Plot the number of unique cusip in the X and y dataframes """

    # define xticks and xticklabels
    xticks = [0]
    years_s0 = [date.split("-")[0] for date in count_df["date"]]
    years_s1 = [date.split("-")[0] for date in count_df["date"].shift(periods=-1).iloc[:-1]]
    for tick, (y_s0, y_s1) in enumerate(zip(years_s0, years_s1)):
        if y_s0 != y_s1:
            xticks.append(tick + 1)
    xticklables = [list(count_df["date"])[_] for _ in xticks]

    # initialize figure
    index = np.arange(len(count_df["date"]))
    fig, ax = plt.subplots(figsize=(16, 10))
    gs = fig.add_gridspec(5, 8)
    ax1 = fig.add_subplot(gs[1:3, :])
    ax2 = fig.add_subplot(gs[3:4, :])
    ax3 = fig.add_subplot(gs[4:5, :])

    # plot the number of stocks
    sns.barplot(x=index, y=count_df["y"], color="#3F00FF", linewidth=0.0, label="y", ax=ax1)
    sns.barplot(x=index, y=count_df["X"], color="green", linewidth=0.0, label="X", ax=ax1)
    sns.barplot(x=index, y=count_df["common"], color="#FF3131", linewidth=0.0, label="common", ax=ax1)
    ax1.set_xticklabels([])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper right")

    # plot the percentage of missing data
    ax2.stem(index, missing_df["stock_perc"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax2.scatter(index, missing_df["stock_perc"].values, color="#899499", marker=".")
    ax2.set_xticklabels([])
    ax2.set_ylabel("Stock missing percentage")

    ax3.stem(index, missing_df["missing_perc"].values, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax3.scatter(index, missing_df["missing_perc"].values, color="#899499", marker=".")
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklables)
    ax3.set_ylabel("Overall missing percentage")

    # save figure
    fig.tight_layout()
    fig.savefig(os.path.join(LOG_PATH, "exploration.pdf"), bbox_inches="tight")
