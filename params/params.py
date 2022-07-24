# window_dict = {
#     "train_win": 240,
#     "valid_win": 240 + 120,
#     "test_win": 240 + 120 + 60,
# }

window_dict = {
    "train_win": 240,
    "valid_win": 240 + 120,
    "test_win": 240 + 120 + 60,
    "resample": True
}

params_dict = {
    "autogluon": {"seq_len": 1, "presets": "medium_quality", "excluded": ["RF"]},  # ["medium", "good", "high", "best"]
    "transformer": {"seq_len": 10, "epochs": 50, "lr": 0.01}
}

horizon_dict = {"horizon": 1}
