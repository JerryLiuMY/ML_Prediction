from global_settings import date0_min, date0_max
from params.params import window_dict
from experiments.experiment import experiment
from experiments.generator import generate_window
from global_settings import OUTPUT_PATH
import os


def run_experiment(model_name, horizons):

    # make directory for the model
    model_path = os.path.join(OUTPUT_PATH, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # perform experiments
    for horizon in horizons:
        # make directory for the horizon
        horizon_path = os.path.join(model_path, f"horizon={horizon}")
        if not os.path.isdir(horizon_path):
            os.mkdir(horizon_path)

        window_gen = generate_window(window_dict, date0_min, date0_max, horizon)
        for window in window_gen:
            experiment(model_name, horizon, window)

    # calculate correlation score
    pass
