from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define directories
DATA_TYPE = "set1_data2"
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    SEAGATE_PATH = "/Volumes/SanDisk_2T/ML_Prediction"
    DATA_PATH = os.path.join(SEAGATE_PATH, DATA_TYPE)
    OUTPUT_PATH = os.path.join(SEAGATE_PATH, "temp")
else:
    DATA_PATH = os.path.join(DESKTOP_PATH, DATA_TYPE)
    OUTPUT_PATH = os.path.join(DESKTOP_PATH, "result")
LOG_PATH = os.path.join(OUTPUT_PATH, "log")


# load necessary files
# trddt_all: intersection between X and y
# cusip_all: intersection between union(X) and union(y)
# cusip_all: cusip_all with match to sic code
trddt_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "trddt_all.pkl")))
cusip_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "cusip_all.pkl")))
num_ft = pd.read_pickle(os.path.join(DATA_PATH, "num_ft.pkl"))
date0_min = "2010-03-31"
date0_max = "2022-06-14"


# TODO
# modify the sampling scheme
# optuna for parameter tuning
# other transformers

# TODO
# US daily alpha / ETF
# Chinese daily alpha / stock index option
# Crypto statistical arbitrage
# Ultra high frequency
