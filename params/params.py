window_dict = {
    "train_win": 240,
    "valid_win": 240 + 120,
    "test_win": 240 + 120 + 60,
}

# ["medium_quality", "good_quality", "high_quality", "best_quality"]
params_dict = {
    "autogluon": {"presets": "medium_quality"}
}
