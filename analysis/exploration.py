import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def plot_cusip():
    """Count for the number of unique cusip in the X and y dataframes"""

    with open(os.path.join(LOG_PATH, "overlap.pkl"), "rb") as f:
        overlap = pickle.load(f)

    cusip_df = pd.DataFrame(columns=["X", "y", "common"], index=overlap)
    for date in overlap:
        X_cusip = list(pd.read_pickle(os.path.join(DATA_PATH, "X", f"{date}.pkl")).index)
        y_cusip = list(pd.read_pickle(os.path.join(DATA_PATH, "y", f"{date}.pkl")).index)
        common_cusip = list(set(X_cusip) & set(y_cusip))
        cusip_df.loc[date, "X"] = len(X_cusip)
        cusip_df.loc[date, "y"] = len(y_cusip)
        cusip_df.loc[date, "common"] = len(common_cusip)
    cusip_df.index.name = "date"
    cusip_df.reset_index(drop=False, inplace=True)
    cusip_df.to_csv(os.path.join(LOG_PATH, "cusip.csv"))

    fig, ax = plt.subplots()
    ax.bar(cusip_df["date"], cusip_df["y"], color="b")
    ax.bar(cusip_df["date"], cusip_df["X"], color="orange")
    ax.bar(cusip_df["date"], cusip_df["common"], color="red")
    ax.set_title("A single plot")
