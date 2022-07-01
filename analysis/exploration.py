import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from global_settings import DATA_PATH
from global_settings import LOG_PATH
import pickle
import glob
import os
sns.set()


def check_overlap():
    """Check for dates overlap in the X and y dataframes"""

    X_dates = []
    for file in glob.glob(os.path.join(DATA_PATH, "X", "*.pkl")):
        name = file.split("/")[-1].split(".")[0]
        if "index" not in name:
            X_dates.append(name)

    y_dates = []
    for file in glob.glob(os.path.join(DATA_PATH, "y", "*.pkl")):
        name = file.split("/")[-1].split(".")[0]
        if "index" not in name:
            y_dates.append(name)

    X_dates, y_dates = sorted(X_dates), sorted(y_dates)
    overlap = sorted(set(X_dates) & set(y_dates))
    X_unique = sorted(set(X_dates) - set(overlap))
    y_unique = sorted(set(y_dates) - set(overlap))

    with open(os.path.join(LOG_PATH, "overlap.pkl"), "wb") as f:
        pickle.dump(overlap, f)

    with open(os.path.join(LOG_PATH, "X_unique.pkl"), "wb") as f:
        pickle.dump(X_unique, f)

    with open(os.path.join(LOG_PATH, "y_unique.pkl"), "wb") as f:
        pickle.dump(y_unique, f)


def build_count(overlap):
    """Count for the number of unique cusip in the X and y dataframes"""

    # build count_df
    count_df = pd.DataFrame(columns=["X", "y", "common"], index=overlap)
    for date in overlap:
        X_cusip = list(pd.read_pickle(os.path.join(DATA_PATH, "X", f"{date}.pkl")).index)
        y_cusip = list(pd.read_pickle(os.path.join(DATA_PATH, "y", f"{date}.pkl")).index)
        common_cusip = list(set(X_cusip) & set(y_cusip))
        count_df.loc[date, "X"] = len(X_cusip)
        count_df.loc[date, "y"] = len(y_cusip)
        count_df.loc[date, "common"] = len(common_cusip)
    count_df.index.name = "date"
    count_df.reset_index(drop=False, inplace=True)
    count_df.to_csv(os.path.join(LOG_PATH, "count.csv"))


def plot_count(count_df):
    """Plot the number of unique cusip in the X and y dataframes"""

    # define xticks and xticklabels
    xticks = [0]
    years_s0 = [date.split("-")[0] for date in count_df["date"]]
    years_s1 = [date.split("-")[0] for date in count_df["date"].shift(periods=-1).iloc[:-1]]
    for tick, (y_s0, y_s1) in enumerate(zip(years_s0, years_s1)):
        if y_s0 != y_s1:
            xticks.append(tick + 1)
    xticklables = [list(count_df["date"])[_] for _ in xticks]

    # plot the number of stocks
    index = np.arange(len(count_df["date"]))
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(x=index, y=count_df["y"], color="#3F00FF", linewidth=0, label="y", ax=ax)
    sns.barplot(x=index, y=count_df["X"], color="green", linewidth=0, label="X", ax=ax)
    sns.barplot(x=index, y=count_df["common"], color="#E34234", linewidth=0, label="common", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklables)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(LOG_PATH, "count.pdf"), bbox_inches="tight")


# if __name__ == "__main__":
#     with open(os.path.join(LOG_PATH, "overlap.pkl"), "rb") as f:
#         overlap = pickle.load(f)
#     build_count(overlap)
#
#
# if __name__ == "__main__":
#     count_df = pd.read_csv(os.path.join(LOG_PATH, "count.csv"))
#     plot_count(count_df)
