# window_dict = {
#     "train_win": 240,
#     "valid_win": 240 + 120,
#     "test_win": 240 + 120 + 60,
# }

window_dict = {
    "train_win": 240,
    "valid_win": 240 + 120,
    "test_win": 240 + 120 + 60,
    "resample": False
}

params_dict = {
    "autogluon": {"presets": "medium_quality", "excluded": ["RF"]}  # ["medium", "good", "high", "best"]
}

horizon_dict = {"horizon": 1}
