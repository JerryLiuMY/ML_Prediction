import os
import numpy as np
import pandas as pd
from pathlib import Path

# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    SEAGATE_PATH = "/Volumes/Sumsung_1T/5_Projects/ML_Prediction"
    DATA_PATH = os.path.join(SEAGATE_PATH, "shared")
    OUTPUT_PATH = os.path.join(SEAGATE_PATH, "result")
else:
    DATA_PATH = os.path.join(DESKTOP_PATH, "shared")
    OUTPUT_PATH = os.path.join(DESKTOP_PATH, "result")
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
# US daily alpha / ETF
# Chinese daily alpha / stock index option
# Crypto statistical arbitrage
# Ultra high frequency

# Reduce validation window
# Sample from training window
# Check GPU usage
