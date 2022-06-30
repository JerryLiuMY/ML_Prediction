import os
import glob
from global_settings import DATA_PATH


def check_overlap():
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

    return overlap, X_unique, y_unique
