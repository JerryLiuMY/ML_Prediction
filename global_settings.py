import os
import numpy as np
import pandas as pd
from pathlib import Path


# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
DATA_PATH = os.path.join(DESKTOP_PATH, "shared")
OUTPUT_PATH = os.path.join(DESKTOP_PATH, "result")
TEMP_PATH = os.path.join(DESKTOP_PATH, "temp")
LOG_PATH = os.path.join(OUTPUT_PATH, "log")

# load necessary files
# trddt_all: intersection between X and y
# cusip_all: intersection between union(X) and union(y)
# cusip_all: cusip_all with match to sic code

trddt_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "trddt_all.pkl")))
cusip_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "cusip_all.pkl")))
cusip_sic = pd.read_csv(os.path.join(DATA_PATH, "cusip_sic.txt"), delim_whitespace=True)
date0_min = "2010-01-04"
date0_max = "2022-06-14"


# TODO
# problems with data
# plot the predictive accuracy for each stock
