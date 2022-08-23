from global_settings import DATA_PATH
import pandas as pd
import pickle5 as pickle
import glob
import os


def build_num_ft():
    """ Find the number of features and save to data directory """

    file_names = [_.split("/")[-1] for _ in glob.glob(os.path.join(DATA_PATH, "X", "*.pkl"))]
    file_names = sorted([_ for _ in file_names if "index" not in _])
    file_name = file_names[0]
    X = pd.read_pickle(os.path.join(DATA_PATH, "X", file_name))
    num_ft = X.shape[1]

    with open(os.path.join(DATA_PATH, "num_ft.pkl"), "wb") as f:
        pickle.dump(num_ft, f, protocol=4)
