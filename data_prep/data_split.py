from datetime import datetime
from global_settings import DATA_PATH
import pickle5 as pickle
import pandas as pd
import os


def split_data():
    # make directories
    X_path = os.path.join(DATA_PATH, "X")
    y_path = os.path.join(DATA_PATH, "y")
    if not os.path.isdir(X_path):
        os.mkdir(X_path)
    if not os.path.isdir(y_path):
        os.mkdir(y_path)

    # load X/y
    X = pd.read_pickle(os.path.join(DATA_PATH, "X.p"))
    y = pd.read_pickle(os.path.join(DATA_PATH, "y.p"))

    # save indices
    X_index = sorted(set(X.index.get_level_values(0)))
    y_index = sorted(set(y.index.get_level_values(0)))
    with open(os.path.join(X_path, "X_index.pkl"), "wb") as f:
        pickle.dump(X_index, f, protocol=4)
    with open(os.path.join(y_path, "y_index.pkl"), "wb") as f:
        pickle.dump(y_index, f, protocol=4)

    # split X
    X_index_0 = X.index.get_level_values(0)
    for X_idx in X_index:
        # adjacent slicing the quickest
        iloc = [i for i, _ in enumerate(X_index_0 == X_idx) if _]
        X_sub = X.iloc[iloc[0]: iloc[-1] + 1].droplevel("date")
        date = datetime.strftime(X_idx, "%Y-%m-%d")
        X_sub.to_pickle(os.path.join(X_path, f"{date}.pkl"), protocol=4)

    # split y
    for y_idx in y_index:
        y_sub = pd.DataFrame(y.loc[y_idx])
        y_sub.columns = ["target"]
        date = datetime.strftime(y_idx, "%Y-%m-%d")
        y_sub = y_sub.dropna(axis=0, how="any", inplace=False)
        y_sub.to_pickle(os.path.join(y_path, f"{date}.pkl"), protocol=4)
