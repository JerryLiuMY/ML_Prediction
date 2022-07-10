window_dict = {
    "train_win": 240,
    "valid_win": 240 + 120,
    "test_win": 240 + 120 + 60,
}

params_dict = {
    "autogluon": {"presets": "high_quality", "excluded": ["RF"]}  # ["medium", "good", "high", "best"]
}

horizon_dict = {"horizon": 1}
