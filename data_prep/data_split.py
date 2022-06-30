import os
import pickle
import pandas as pd
from datetime import datetime
from global_settings import DATA_PATH


# load X/y and define indices
X = pd.read_pickle(os.path.join(DATA_PATH, "X.p"))
y = pd.read_pickle(os.path.join(DATA_PATH, "y.p"))
y_index = sorted(set(y.index.get_level_values(0)))
X_index = sorted(set(X.index.get_level_values(0)))
X_path = os.path.join(DATA_PATH, "X")
y_path = os.path.join(DATA_PATH, "y")

# make directories
if not os.path.isdir(X_path):
    os.mkdir(X_path)

if not os.path.isdir(y_path):
    os.mkdir(y_path)


# save indices
def save_indices():
    """Save indices for X and y"""
    with open(os.path.join(X_path, "X_index.pkl"), 'wb') as f:
        pickle.dump(X_index, f)

    with open(os.path.join(y_path, "y_index.pkl"), 'wb') as f:
        pickle.dump(y_index, f)


# split X
def split_X():
    """Split X based on date"""
    for X_idx in X_index:
        date = datetime.strftime(X_idx, "%Y-%m-%d")
        X_sub = X.loc[X_idx]
        X_sub.to_pickle(os.path.join(X_path, f"{date}.pkl"))


# split y
def split_y():
    """Split y based on date"""
    for y_idx in y_index:
        date = datetime.strftime(y_idx, "%Y-%m-%d")
        y_sub = y.loc[y_idx]
        y_sub.to_pickle(os.path.join(y_path, f"{date}.pkl"))
