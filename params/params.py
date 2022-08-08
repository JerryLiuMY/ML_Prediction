# window_dict = {
#     "train_win": 240,
#     "valid_win": 240 + 120,
#     "test_win": 240 + 120 + 60,
# }

data_dict = {
    "train_win": 960,
    "valid_win": 960 + 120,
    "test_win": 960 + 120 + 60,
    "resample": True,
    "imputation": "default"  # ["default", "zero", "drop"]
}

params_dict = {
    "autogluon": {"seq_len": 1, "presets": "medium_quality", "excluded": ["RF"]},  # ["medium", "good", "high", "best"]
    "transformer": {"seq_len": 5, "nlayer": 2, "nhead": 6, "d_model": 6144, "dropout": 0.1, "epochs": 10, "lr": 0.01}
}

horizon_dict = {"horizon": 1}
