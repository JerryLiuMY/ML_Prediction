# window_dict = {
#     "train_win": 240,
#     "valid_win": 240 + 120,
#     "test_win": 240 + 120 + 60,
# }

data_dict = {
    "train_win": 240,
    "valid_win": 240 + 120,
    "test_win": 240 + 120 + 60,
    "resample": True,
    "imputation": "zero"  # ["default", "zero", "drop"]
}

params_dict = {
    "autogluon": {"seq_len": 1, "presets": "medium_quality", "excluded": ["RF"]},  # ["medium", "good", "high", "best"]
    "transformer": {"seq_len": 5, "nlayer": 3, "nhead": 8, "d_model": 8192, "dropout": 0.1, "epochs": 10, "lr": 0.01}
}

horizon_dict = {"horizon": 1}
