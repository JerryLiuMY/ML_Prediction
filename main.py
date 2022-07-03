from global_settings import date0_min, date0_max
from params.params import window_dict
from experiments.experiment import experiment
from experiments.generator import generate_window
from experiments.summary import summarize
from global_settings import OUTPUT_PATH
import os


def run_experiment(model_name, horizons):
    """ Perform experiment for a particular model
    :param model_name: model name
    :param horizons: list of horizons
    """

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

        # generate windows
        window_gen = generate_window(window_dict, date0_min, date0_max, horizon)
        for window in window_gen:
            # make directory for the window
            window_path = os.path.join(horizon_path, window["X"][0][0])
            if not os.path.isdir(window_path):
                os.mkdir(window_path)
                os.mkdir(os.path.join(window_path, "info"))
                os.mkdir(os.path.join(window_path, "predict"))
                os.mkdir(os.path.join(window_path, "summary"))
                experiment(model_name, horizon, window)
                summarize(model_name, horizon, window)


if __name__ == "__main__":
    model_name = "autogluon"
    horizons = [1, 2, 3, 4, 5, 10, 20, 30, 50]
    run_experiment(model_name, horizons)
