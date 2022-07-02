import os
import numpy as np
import pandas as pd
from pathlib import Path


# directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
DATA_PATH = os.path.join(DESKTOP_PATH, "shared")
OUTPUT_PATH = os.path.join(DESKTOP_PATH, "result")
TEMP_PATH = os.path.join(DESKTOP_PATH, "temp")
LOG_PATH = os.path.join(OUTPUT_PATH, "log")

trddt_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "trddt_all.pkl")))
date0_min = "2010-01-04"
date0_max = "2022-06-14"


# TODO
# AutoGluon

# window size
# horizons

# problems with data
# count number of X
# count number of y
# count number of overlaps
# plot the predictive accuracy for each stock
