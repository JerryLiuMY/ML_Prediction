from global_settings import DATA_PATH
from tqdm.notebook import tqdm
import pandas as pd
import pickle5 as pickle
import glob
import os


def build_cusip():
    """ Build cusip information """

    # union of cusip for X
    cusip_X = set()
    file_names = [_.split("/")[-1] for _ in glob.glob(os.path.join(DATA_PATH, "X", "*.pkl"))]
    file_names = sorted([_ for _ in file_names if "index" not in _])
    for file_name in tqdm(file_names):
        X = pd.read_pickle(os.path.join(DATA_PATH, "X", file_name))
        cusip_X = cusip_X.union(set(X.index))

    # union of cusip for y
    cusip_y = set()
    file_names = [_.split("/")[-1] for _ in glob.glob(os.path.join(DATA_PATH, "y", "*.pkl"))]
    file_names = sorted([_ for _ in file_names if "index" not in _])
    for file_name in tqdm(file_names):
        y = pd.read_pickle(os.path.join(DATA_PATH, "y", file_name))
        cusip_y = cusip_y.union(set(y.index))

    # find intersection
    cusip_all = list(cusip_X & cusip_y)
    with open(os.path.join(DATA_PATH, "cusip_all.pkl"), "wb") as f:
        pickle.dump(cusip_all, f, protocol=4)
